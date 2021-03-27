import pandas as pd

all_series_lines = pd.read_json('all_scripts_raw.json')

def remove_speakers_and_empty_lines(episode_content: str) -> str:
    """Removes superfluous empty lines and the names of the speakers from the input data:
    e.g. :
    Picard: Make it so.
    becomes:
    Make it so.
    """

    cleaned_lines = []
    for line in episode_content.split('\n'):
        if line == '':
            continue
        if ':' in line:
            part_to_keep = line.split(':')[1]
            cleaned_lines.append(part_to_keep.strip() + ' \n')
            continue
        
        
        cleaned_lines.append(line.strip() + ' \n')
    return ''.join(cleaned_lines)



#  remove the names of the speakers and get rid of the empty lines
# and I'll focus on The Next Generation Episodes for now
tng_series_lines_cleaned = all_series_lines.TNG.map(remove_speakers_and_empty_lines)
print(tng_series_lines_cleaned)



