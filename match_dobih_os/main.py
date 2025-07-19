# Load DoBIH

import json
import os
from pprint import pprint
import re
import shutil
import time
import zipfile

import anthropic
import geopandas
from jellyfish import levenshtein_distance
import pandas as pd

from .config import PROJECT_DIR
from .helpers import download, force_ascii


CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_matches.json")

# Initialize Claude client
client = anthropic.Anthropic()

# Token tracking (per-run limit)
TOKEN_LIMIT = 250000
tokens_used = 0
llm_disabled = False


dobih_csv_path = os.path.join(CACHE_DIR, "dobih.csv")
if not os.path.exists(dobih_csv_path):
    dobih_zip_path = os.path.join(CACHE_DIR, "hillcsv.zip")
    download("https://hills-database.co.uk/hillcsv.zip", dobih_zip_path)
    with zipfile.ZipFile(dobih_zip_path) as zc:
        zf_contents = zc.infolist()
        assert len(zf_contents) == 1
        with zc.open(zf_contents[0]) as in_f, open(dobih_csv_path, "wb") as out_f:
            shutil.copyfileobj(in_f, out_f)
dobih_hills_raw = pd.read_csv(
    dobih_csv_path,
    usecols=[
        "Number",
        "Name",
        "Metres",
        "Xcoord",
        "Ycoord",
        "Country",
        "Streetmap/MountainViews",
    ],
    dtype={
        "Name": "string",
        "Country": "string",
    },
)
dobih_hills = geopandas.GeoDataFrame(
    dobih_hills_raw,
    geometry=geopandas.points_from_xy(dobih_hills_raw.Xcoord, dobih_hills_raw.Ycoord),
    crs="EPSG:27700",
)

os_names_path = os.path.join(CACHE_DIR, "os_names.gpkg")
if not os.path.exists(os_names_path):
    os_names_zip_path = os.path.join(CACHE_DIR, "os_names.zip")
    download(
        "https://api.os.uk/downloads/v1/products/OpenNames/downloads?area=GB&format=GeoPackage&redirect",
        os_names_zip_path,
    )
    with zipfile.ZipFile(os_names_zip_path) as zc:
        with zc.open("Data/opname_gb.gpkg") as in_f, open(os_names_path, "wb") as out_f:
            shutil.copyfileobj(in_f, out_f)
os_names = geopandas.read_file(
    os_names_path,
    columns=["id", "name1", "name2"],
    where="local_type = 'Hill Or Mountain'",
)


def dobih_name_variants(name: str):
    names = []

    name = name.lower()

    # remove qualifiers. e.g.: "Meall nan Gabhar (old GR)"
    name = re.sub(r" \(.*?\)", "", name)

    alternative_pat = r" \[(.*?)\]"
    names += re.findall(alternative_pat, name)
    name = re.sub(alternative_pat, "", name)

    names.append(name)
    if " - " in name:
        names += name.split(" - ")

    out = []
    for n in names:
        out.append(
            re.sub(
                r"[ -]",
                "",
                re.sub(r" (north|n|south|s|east|e|west|w|ne|nw|se|sw) top$", "", n),
            )
        )
        out.append(re.sub(r"[ -]", "", n))
    return out

def os_name_variants(r):
    return [
        re.sub(r"[ -]", "", force_ascii(n).lower())
        for n in ([r.name1, r.name2] if r.name2 is not None else [r.name1])
    ]


def load_llm_cache():
    """Load cached LLM responses."""
    if os.path.exists(LLM_CACHE_FILE):
        with open(LLM_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_llm_cache(cache):
    """Save LLM cache to file."""
    os.makedirs(os.path.dirname(LLM_CACHE_FILE), exist_ok=True)
    with open(LLM_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def llm_match_hills(dobih_name, os_candidates):
    """Use Claude to match DOBIH hill to OS candidates."""
    global tokens_used, llm_disabled

    cache = load_llm_cache()

    # Create cache key
    os_names = [
        f"{row.name1}" + (f" ({row.name2})" if row.name2 else "")
        for _, row in os_candidates.iterrows()
    ]
    cache_key = f"{dobih_name}|||{';'.join(sorted(os_names))}"

    if cache_key in cache:
        return cache[cache_key]

    # Check if LLM is disabled due to token limit (after checking cache)
    if llm_disabled:
        return None

    # Create OS candidates list for prompt
    os_list = []
    for i, (_, row) in enumerate(os_candidates.iterrows()):
        name = row.name1
        if row.name2:
            name += f" (also known as {row.name2})"
        os_list.append(f"{i+1}. {name}")

    prompt = f"""You are an expert geographic information system analyst tasked with matching hills between two databases: the Database of British and Irish Hills (DOBIH) and Ordnance Survey Open Names (OS). Your goal is to find the OS hill that matches the given DOBIH hill, if any.

Here is the DOBIH hill name you need to match:
<dobih_hill_name>
{dobih_name}
</dobih_hill_name>

Here is the list of OS candidate hills in the area:
<os_candidate_hills>
{chr(10).join(os_list)}
</os_candidate_hills>

Your task is to determine which OS hill (if any) matches the DOBIH hill. Follow these steps:

1. Compare the DOBIH hill name to each OS candidate hill name.
2. Consider the following factors:
   - Different spellings of the same hill name
   - Alternative names for the same geographical feature
   - Ensure that the hills represent the same peak or feature (reject if they are different)

3. In <comparison_analysis> tags, provide your thought process for each comparison. For each OS candidate hill:
   a. List the candidate with a number prefix (e.g., 1. [Hill Name])
   b. Compare the spelling and potential alternative names
   c. Consider geographical context if available
   d. Rate the similarity on a scale of 1-5 (1 being least similar, 5 being most similar)
   e. Provide a brief explanation for the rating

4. After analyzing all candidates, identify the best match (if any) and explain why.

5. After your analysis, provide your final answer in <result> tags. The result should be ONLY one of the following:
   - The number of the matching OS hill (a number from 1 to {len(os_list)})
   - The word "none" if no match is found

Important: Your final output in the <result> tags must contain ONLY the number or "none", with no additional explanation or reasoning.

Example output structure (do not use this content, it's just to illustrate the format):

<comparison_analysis>
1. [OS Hill Name 1]
   Similarity: 3/5
   Explanation: Similar spelling, but potentially different feature.

2. [OS Hill Name 2]
   Similarity: 5/5
   Explanation: Exact match in spelling and likely the same feature.

[... continue for all candidates ...]

Best match: Candidate 2, because it's an exact match in spelling and likely represents the same geographical feature.
</comparison_analysis>

<result>
2
</result>

OR

<result>
none
</result>

Please proceed with your analysis and provide the result."""

    print(f"LLM Query: Matching DOBIH '{dobih_name}' against OS candidates: {", ".join(os_list)}")
    
    # Retry logic with exponential backoff
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            
            # Track token usage
            tokens_used += response.usage.input_tokens + response.usage.output_tokens

            # Check if we've exceeded the limit
            if tokens_used > TOKEN_LIMIT:
                llm_disabled = True
                print(
                    f"WARNING: Token limit exceeded ({tokens_used:,} > {TOKEN_LIMIT:,}). Disabling LLM matching for remaining hills."
                )
                return None

            result_text = response.content[0].text.strip()
            
            # Extract result from <result> tags
            import re
            result_match = re.search(r'<result>\s*(.*?)\s*</result>', result_text, re.DOTALL | re.IGNORECASE)
            if result_match:
                result = result_match.group(1).strip().lower()
            else:
                raise RuntimeError("No result tags in response: "+result_text)

            # Parse response
            if result == "none":
                match_idx = None
            else:
                try:
                    match_idx = int(result) - 1  # Convert to 0-based index
                    if match_idx < 0 or match_idx >= len(os_candidates):
                        match_idx = None
                except ValueError:
                    raise RuntimeError("Non-numeric result: "+result_text)

            # Cache the result
            cache[cache_key] = match_idx
            save_llm_cache(cache)
            
            print(f"-> Match: {match_idx}")
            return match_idx
            
        except anthropic.APIError as e:
            error_type = type(e).__name__
            print(f"Anthropic {error_type} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for {error_type}. Skipping this match.")
                return None

    return None


def match_one(dobih_row):
    os_candidates = os_names.take(
        os_names.sindex.query(dobih_row.geometry, predicate="dwithin", distance=150)
    )
    if len(os_candidates) == 0:
        return None

    distances = os_candidates.geometry.distance(dobih_row.geometry)

    best_name_similarity = None
    best_os_i = None

    for i, os_row in os_candidates.iterrows():
        os_name_vs = os_name_variants(os_row)
        dobih_name_vs = dobih_name_variants(dobih_row.Name)

        name_similarity = 0
        for os_name in os_name_vs:
            for dobih_name in dobih_name_vs:
                if len(os_name) == 0 or len(dobih_name) == 0:
                    continue

                # Normalized Levenshtein: 1 - (distance / max_length)
                dist = levenshtein_distance(os_name, dobih_name)
                max_len = max(len(os_name), len(dobih_name))
                similarity = 1 - (dist / max_len)
                name_similarity = max(name_similarity, similarity)

        if best_name_similarity == None or name_similarity > best_name_similarity:
            best_name_similarity = name_similarity
            best_os_i = i

    if best_name_similarity is None or best_name_similarity < 0.3:
        return None

    if best_name_similarity >= 0.9:
        return {
            "os_row": os_candidates.loc[best_os_i],
            "distance": distances.loc[best_os_i],
            "name_similarity": best_name_similarity,
        }

    # Use LLM for ambiguous cases
    llm_match_idx = llm_match_hills(dobih_row.Name, os_candidates)
    if llm_match_idx is not None:
        matched_row = os_candidates.iloc[llm_match_idx]
        return {
            "os_row": matched_row,
            "distance": distances.loc[matched_row.name],
            "name_similarity": -1,  # Indicate LLM match
        }

    return None


def main():
    results = []

    for _, dobih_row in dobih_hills.iterrows():
        match_result = match_one(dobih_row)

        if match_result is None:
            continue

        os_row = match_result["os_row"]

        results.append(
            {
                "dobih_number": dobih_row.Number,
                "dobih_name": dobih_row.Name,
                "os_id": os_row.id,
                "os_name": os_row.name1,
                "distance": match_result["distance"],
                "name_similarity": match_result["name_similarity"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv("debug_results.csv", index=False)
    print(
        f"Saved {len(results_df)} matches (of {len(dobih_hills)} dobih hills) to debug_results.csv"
    )

    simplified_df = results_df[
        ["dobih_number", "os_id", "dobih_name", "os_name"]
    ].copy()
    simplified_df.to_csv("match_dobih_os.csv", index=False)
    print(f"Saved match_dobih_os.csv ({len(simplified_df)} rows of data)")
