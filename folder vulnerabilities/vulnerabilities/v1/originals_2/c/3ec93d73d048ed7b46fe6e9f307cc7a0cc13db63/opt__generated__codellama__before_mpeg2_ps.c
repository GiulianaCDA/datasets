[PYTHON]
def get_closest_video_frame(timestamp: int, video_frames: list) -> tuple:
    """
    Returns the timestamp of the closest frame before or equal to the given timestamp.

    :param timestamp: Timestamp in milliseconds
    :type timestamp: int
    :param video_frames: List containing the timestamps of all frames
    :type video_frames: list
    :return: Tuple with the index of the closest frame and its timestamp
    :rtype: tuple
    """
    # Find the first frame after the given timestamp
    closest_frame = next(filter(lambda x: x >= timestamp, video_frames), None)

    if closest_frame is None:
        # No frames with a timestamp larger than the given one, return the last frame
        return len(video_frames) - 1, video_frames[-1]
    else:
        # Return the index of the frame and its timestamp
        return video_frames.index(closest_frame), closest_frame

def get_closest_audio_frame(timestamp: int, audio_frames: list) -> tuple:
    """
    Returns the timestamp of the closest frame before or equal to the given timestamp.

    :param timestamp: Timestamp in milliseconds
    :type timestamp: int
    :param audio_frames: List containing the timestamps of all frames
    :type audio_frames: list
    :return: Tuple with the index of the closest frame and its timestamp
    :rtype: tuple
    """
    # Find the first frame after the given timestamp
    closest_frame = next(filter(lambda x: x >= timestamp, audio_frames), None)

    if closest_frame is None:
        # No frames with a timestamp larger than the given one, return the last frame
        return len(audio_frames) - 1, audio_frames[-1]
    else:
        # Return the index of the frame and its timestamp
        return audio_frames.index(closest_frame), closest_frame

def get_timestamp(video_timestamps: list, audio_timestamps: list) -> tuple:
    """
    Returns the timestamp that is closest to the given timestamp.

    :param video_timestamps: List containing the timestamps of all frames in the video stream
    :type video_timestamps: list
    :param audio_timestamps: List containing the timestamps of all frames in the audio stream
    :type audio_timestamps: list
    :return: Tuple with the index of the closest frame and its timestamp
    :rtype: tuple
    """
    # Get the closest video frame and audio frame
    video_index, video_timestamp = get_closest_video_frame(10000, video_timestamps)
    audio_index, audio_timestamp = get_closest_audio_frame(10000, audio_timestamps)

    # Calculate the difference between both frames
    diff = abs(video_timestamp - audio_timestamp)

    # Return the index of the closest frame and its timestamp
    if diff == 0:
        return video_index, video_timestamp
    elif diff > 10000:
        return audio_index, audio_timestamp
    else:
        return video_index, video_timestamp
[/PYTHON]
