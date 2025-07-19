# match-dobih-os

This tool matches hills between the Database of British and Irish Hills (DoBIH) and Ordnance Survey Open Names (OS).

**📊 [match_dobih_os.csv](https://github.com/dzfranklin/match-dobih-os/blob/main/match_dobih_os.csv)**

## Algorithm

For each DoBIH Hill:
1. Find OS hill names within 150m
2. Normalizes hill names and split out variants
3. Look for a match where a normalized variant of the DoBIH name matches a normalized variant of the OS name
    - if there is a candidate with similarity > 0.9: accept it (~50%)
    - if all candidates have similarity < 0.3: reject it (~50%)
    - otherwise: ask an LLM (<1%)

## Results

The algorithm successfully matched **11,522 hills** out of 21,547.

### Examples

**Exact matches** (exact names, 0m distance):
- 'Beinn Dearg', 'Torlum', 'Ben Ledi'

**Furthest matches** (>140m, exact names):
- 'Dunearn Hill' (149.8m), 'Mulfran' (149.1m)

**Complex LLM resolutions** (different starting letters):
- 'Bhasteir Tooth [Basteir Tooth]' → 'Am Bàsteir'
- 'Skiddaw Little Man' → 'Little Man'

**Duplicate OS mappings** (one OS hill matching multiple DoBIH entries):
- OS 'Am Bàsteir' matches: 'Am Basteir' + 'Bhasteir Tooth [Basteir Tooth]'
- OS 'The Saddle' matches: 'The Saddle - Trig Point' + 'The Saddle'

I have no affiliation with DoBIH or OS.
