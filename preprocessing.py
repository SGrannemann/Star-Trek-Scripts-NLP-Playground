import pandas as pd



def remove_speakers_and_empty_lines(episode_content: str) -> str:
    """Removes superfluous empty lines and the names of the speakers from the input data:
    e.g. :
    Picard: Make it so.
    becomes:
    Make it so.
    """

    cleaned_lines = []
    for line in episode_content.split('\n'):
        # ignore empty lines
        if line == '':
            continue
        # the actual talking lines always contain a ':' - we will just keep the text, not the talker
        # for this application
        if ':' in line:
            part_to_keep = line.split(':')[1]
            cleaned_lines.append(part_to_keep.strip() + ' \n')
            continue
        # after this string there are only information about the franchise, we can leave those out.
        if line == '<Back':
            break
        
        
        cleaned_lines.append(line.strip() + ' \n')
    return ''.join(cleaned_lines)


all_series_scripts = pd.read_json('all_scripts_raw.json')
#  remove the names of the speakers and get rid of the empty lines
# and I'll focus on The Next Generation Episodes for now
tng_series_scripts_cleaned = all_series_scripts.TNG.map(remove_speakers_and_empty_lines)

# after this preprocessing



