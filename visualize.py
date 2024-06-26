import os

import matplotlib.pyplot as plt
from PIL import Image
from file_manager import FileManager

import warnings
warnings.filterwarnings("ignore")

def visualize_event_traces(dataframe, xes_file, LoA, stride=0.5):
    activities = dataframe['concept:name'].unique()
    activity_colors = {activity: plt.cm.nipy_spectral(i / len(activities) + stride) for i, activity in
                       enumerate(activities)}

    fig, ax = plt.subplots(figsize=(15, 5))

    for i, (case_id, group) in enumerate(dataframe.groupby('case:concept:name')):
        group = group.reset_index(drop=True)
        for j in range(len(group)):
            ax.barh(i, 1, left=j, color=activity_colors[group.at[j, 'concept:name']], edgecolor='none')

    ax.set_yticks(range(i + 1))
    ax.set_yticklabels(dataframe['case:concept:name'].unique())
    ax.set_xlabel('Events in Trace')
    ax.set_title('Event Trace Visualization')
    plt.gca().invert_yaxis()

    base_dir = "plots"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    filename = FileManager.get_filename(xes_file)
    default_filename = f"{filename}_LoA_{LoA}.png"
    output_path = FileManager.get_save_path(default_filename, ".png")

    if not output_path:
        output_path = os.path.join('plots', default_filename)
        print(f"Saving cancelled, saved to default {base_dir} dir.")

    plt.savefig(output_path)
    plt.close(fig)

    img = Image.open(output_path)
    img.show()
