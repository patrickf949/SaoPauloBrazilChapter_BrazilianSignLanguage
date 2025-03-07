# we can use this code to:
# - Download all videos for one word from one data source 
#   - (for review during the data organisation and cleaning)
# - Download all videos for one word from all data sources 
#   - (for review during the data organisation and cleaning)
# - Download all videos for a collection of words from all data source 
#   - (for creating our raw combined dataset, after we have decided our target words)

import requests
import pandas as pd


def download_video_from_link(
    link: str,
    output_path: str,
) -> None:
    """
    Download a video from a URL and save it to a file.
    """
    pass

def download_video_from_metadata(
    metadata: pd.DataFrame,
    data_source: str,
    output_path: str,
) -> None:
    """
    Download all videos for one word from one data source.
    """
    pass

