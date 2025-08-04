import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),  # left eye and ear
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),  # right eye and ear
    (9, 10),  # mouth
    (11, 12),  # shoulders
    (11, 13),
    (13, 15),  # left arm
    (12, 14),
    (14, 16),  # right arm
    (15, 17),
    (15, 19),
    (15, 21),  # left wrist to fingers
    (16, 18),
    (16, 20),
    (16, 22),  # right wrist to fingers
    (11, 23),
    (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),  # left leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),  # right leg
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index Finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle Finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring Finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky Finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def draw_angle_arc(ax, a, b, c, color="orange", radius=0.05, linewidth=1):
    # Vectors BA and BC
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # # Get angle in radians and degrees
    # angle_rad = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
    # angle_deg = np.degrees(angle_rad)

    # Calculate start and end angles for the arc (in degrees)
    start_angle = np.degrees(np.arctan2(v1_norm[1], v1_norm[0]))
    end_angle = np.degrees(np.arctan2(v2_norm[1], v2_norm[0]))

    # Adjust angles so that the arc is drawn in the correct direction
    # (matplotlib always goes counterclockwise)
    if end_angle < start_angle:
        end_angle += 360

    arc = patches.Arc(
        b,
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=start_angle,
        theta2=end_angle,
        color=color,
        linewidth=linewidth,
    )
    ax.add_patch(arc)


def visualize_landmark_features(
    landmarks: List,
    landmark_type: str,
    distances: List[Tuple[int, int]] = None,
    angles: List[Tuple[int, int, int]] = None,
    title: str = "Landmark Feature Visualization",
    show_legs: bool = True,
    show_waist: bool = True,
    annotation_fontsize: int = 8,
    annotation_offset: Tuple[float, float] = (0.01, 0.01),
    annotation_fontweight: str = "normal",
    point_color: str = "blue",
    connection_color: str = "blue",
    connection_style: str = "-",
    connection_width: float = 1.0,
    annotation_color: str = "red",
    distance_color: str = "green",
    distance_style: str = "--",
    distance_width: float = 2.0,
    angle_color: str = "orange",
    angle_width: float = 1.0,
    background_color: str = "white",
):
    """
    Visualizes a set of landmarks with optional skeleton, distances, and angles.

    Parameters:
    -----------
    landmarks : mp.framework.formats.landmark_pb2.NormalizedLandmarkList
        The list of MediaPipe landmarks to visualize. Each landmark should have `x` and `y` attributes.

    landmark_type : str
        Either 'pose' or 'hand'. Determines which default skeleton connections to draw.

    distances : List[Tuple[int, int]], optional
        A list of (start_index, end_index) pairs representing custom distances to draw as green lines.

    angles : List[Tuple[int, int, int]], optional
        A list of (point_a, point_b, point_c) triplets representing angle connections (V-shapes), drawn in orange.
        Each triplet defines an angle at point_b between segments AB and CB.

    title : str
        Title of the plot.

    show_legs : bool
        Whether to show leg landmarks and connections (points 23-32).

    show_waist : bool
        Whether to show waist landmarks and connections (points 23-24 and connections to shoulders).

    annotation_fontsize : int, optional
        Font size for the landmark index annotations (default: 8).

    annotation_offset : Tuple[float, float], optional
        Offset (dx, dy) for annotation text placement relative to landmark points (default: (0.01, 0.01)).
        Positive values move text right and down.

    annotation_fontweight : str, optional
        Font weight for the landmark annotations. Can be one of: 'normal', 'bold', 'light', etc. (default: 'normal').

    point_color : str, optional
        Color of the landmark points (default: 'blue').

    connection_color : str, optional
        Color of the lines connecting landmarks (default: 'blue').

    connection_style : str, optional
        Style of the connection lines, e.g. '-', '--', ':' (default: '-').

    connection_width : float, optional
        Width of the connection lines (default: 1.0).

    annotation_color : str, optional
        Color of the landmark index annotations (default: 'red').

    distance_color : str, optional
        Color of the custom distance lines (default: 'green').

    distance_style : str, optional
        Style of the distance lines, e.g. '-', '--', ':' (default: '--').

    distance_width : float, optional
        Width of the distance lines (default: 2.0).

    angle_color : str, optional
        Color of the angle arcs (default: 'orange').

    angle_width : float, optional
        Width of the angle arc lines (default: 1.0).

    background_color : str, optional
        Color of the plot background (default: 'white').

    Notes:
    ------
    - The function automatically inverts the Y-axis to match image coordinates.
    - Uses predefined `POSE_CONNECTIONS` and `HAND_CONNECTIONS` for skeleton drawing. These should be defined globally.
    - Landmark indices are shown in red for easy identification.

    Returns:
    --------
    None. Displays the plot.
    """
    assert landmark_type in ["hand", "pose"], "landmark_type must be 'hand' or 'pose'"

    landmarks = landmarks.landmark
    x = [lm.x for lm in landmarks]
    y = [lm.y for lm in landmarks]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_facecolor(background_color)
    
    # Filter which points to display based on show_legs and show_waist
    if landmark_type == "pose":
        visible_points = set(range(len(landmarks)))
        if not show_legs:
            # Remove leg points (25-32) but keep waist points (23-24)
            for i in range(25, 33):
                visible_points.discard(i)
        if not show_waist:
            # Remove only waist points (23-24)
            visible_points.discard(23)
            visible_points.discard(24)
        
    # Draw main skeleton (bottom layer, zorder=1)
    if landmark_type == "pose":
        connections = POSE_CONNECTIONS
        radius = 0.05
        
        # Filter connections based on show_legs and show_waist
        filtered_connections = []
        for start, end in connections:
            # Skip leg connections (25-32)
            if not show_legs and (start >= 25 or end >= 25):
                continue
            # Skip waist connections
            if not show_waist and ((start == 23 and end == 24) or  # hip connection
                                ((start in [11, 12] and end in [23, 24]) or  # shoulder to hip
                                 (end in [11, 12] and start in [23, 24]))):  # hip to shoulder
                continue
            filtered_connections.append((start, end))
        
        connections = filtered_connections
    else:
        connections = HAND_CONNECTIONS
        radius = 0.005

    # Draw connections (bottom layer)
    for start, end in connections:
        plt.plot([x[start], x[end]], [y[start], y[end]], 
                color=connection_color, 
                linestyle=connection_style, 
                linewidth=connection_width,
                zorder=1)

    # Plot points (middle layer)
    if landmark_type == "pose":
        visible_x = [x[i] for i in visible_points]
        visible_y = [y[i] for i in visible_points]
        plt.scatter(visible_x, visible_y, c=point_color, edgecolors=connection_color, zorder=2)
    else:
        plt.scatter(x, y, c=point_color, edgecolors=connection_color, zorder=2)

    # Draw distances (upper middle layer)
    if distances:
        for start, end in distances:
            # Skip if either point is hidden
            if landmark_type == "pose" and (start not in visible_points or end not in visible_points):
                continue
            # Draw line with arrowheads at both start and end using annotate
            ax = plt.gca()
            ax.annotate(
                '', 
                xy=(x[end], y[end]), 
                xytext=(x[start], y[start]),
                arrowprops=dict(
                    arrowstyle='<|-|>',  # double-headed hollow arrow
                    color=distance_color,
                    linestyle=distance_style,
                    linewidth=distance_width,
                    shrinkA=0, shrinkB=0,
                    fill=False  # ensure hollow arrowhead
                ),
                zorder=3
            )
            

    # Draw angle connections (upper middle layer)
    if angles:
        ax = plt.gca()  # get current axis
        for a_idx, b_idx, c_idx in angles:
            # Skip if any point is hidden
            if landmark_type == "pose" and not all(idx in visible_points for idx in [a_idx, b_idx, c_idx]):
                continue
            a = (x[a_idx], y[a_idx])
            b = (x[b_idx], y[b_idx])
            c = (x[c_idx], y[c_idx])

            # Create arc with zorder
            arc = patches.Arc(
                b,
                width=2 * radius,
                height=2 * radius,
                angle=0,
                theta1=np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0])),
                theta2=np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0])),
                color=angle_color,
                linewidth=angle_width,
                zorder=3
            )
            
            ax.add_patch(arc)

    # Landmark indices (top layer)
    dx, dy = annotation_offset
    # Create path effects for text outline
    text_outline = [
        path_effects.Stroke(linewidth=3, foreground=(.9,.9,.9)),  # Thick grey edge
        path_effects.Normal()  # Normal text on top
    ]
    
    if landmark_type == "pose":
        for i in visible_points:
            plt.text(x[i] + dx, y[i] + dy, str(i), 
                    fontsize=annotation_fontsize, 
                    fontweight=annotation_fontweight,
                    color=annotation_color,
                    zorder=4,
                    path_effects=text_outline)
    else:
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi + dx, yi + dy, str(i), 
                    fontsize=annotation_fontsize,
                    fontweight=annotation_fontweight,
                    color=annotation_color,
                    zorder=4,
                    path_effects=text_outline)

    plt.xticks([])
    plt.yticks([])
    
    plt.gca().invert_yaxis()
    plt.title(title, fontsize = 16)
    plt.axis("equal")
    
    # Create legend elements
    legend_elements = [
        # Always add landmark points to legend
        plt.Line2D([0], [0], 
                  marker='o', 
                  color=connection_color,
                  markerfacecolor=point_color,
                  markersize=6,
                  linestyle='none',
                  label="Landmarks")
    ]
    
    # Add distance line to legend if distances are plotted
    if distances:
        legend_elements.append(plt.Line2D([0], [0], 
                                        color=distance_color,
                                        linestyle=distance_style,
                                        linewidth=distance_width,
                                        label="Distance"))
    
    # Add angle representation to legend if angles are plotted
    if angles:
        legend_elements.append(plt.Line2D([0], [0],
                                        marker='o',
                                        color=angle_color,
                                        markerfacecolor='none',
                                        markersize=10,
                                        linestyle='none',
                                        label="Angle"))
    
    # Always show legend since we always have landmark points
    plt.legend(handles=legend_elements,
              framealpha=1)
    plt.tight_layout()

def visualize_differences(
    prev_landmarks: List,
    next_landmarks: List,
    landmark_type: str,
    landmark_differences: List[int],
    title: str = "2D Landmark Differences",
    show_legs: bool = True,
    show_waist: bool = True,
    prev_point_color: str = "blue",
    prev_connection_color: str = "blue",
    prev_connection_style: str = "--",
    prev_connection_width: float = 1.0,
    next_point_color: str = "green",
    next_connection_color: str = "green",
    next_connection_style: str = "--",
    next_connection_width: float = 1.0,
    arrow_color: str = "red",
    arrow_alpha: float = 0.6,
    arrow_width: float = 0.01,
    background_color: str = "white",
):
    """
    Visualizes motion between two 2D landmark frames using red arrows.

    Parameters:
    -----------
    prev_landmarks : List of landmarks (with x, y)
        Previous frame landmarks.

    next_landmarks : List of landmarks (with x, y)
        Current frame landmarks.

    landmark_type : str
        Either 'pose' or 'hand'. Determines which default skeleton connections to draw.

    landmarks_differences: List of landmarks

    title : str
        Title of the plot.

    show_legs : bool
        Whether to show leg landmarks and connections (points 23-32).

    show_waist : bool
        Whether to show waist landmarks and connections (points 23-24 and connections to shoulders).

    prev_point_color : str, optional
        Color of the previous frame's landmark points (default: 'blue').

    prev_connection_color : str, optional
        Color of the previous frame's connection lines (default: 'blue').

    prev_connection_style : str, optional
        Style of the previous frame's connection lines (default: '--').

    prev_connection_width : float, optional
        Width of the previous frame's connection lines (default: 1.0).

    next_point_color : str, optional
        Color of the next frame's landmark points (default: 'green').

    next_connection_color : str, optional
        Color of the next frame's connection lines (default: 'green').

    next_connection_style : str, optional
        Style of the next frame's connection lines (default: '--').

    next_connection_width : float, optional
        Width of the next frame's connection lines (default: 1.0).

    arrow_color : str, optional
        Color of the motion arrows between frames (default: 'red').

    arrow_alpha : float, optional
        Transparency of the motion arrows (default: 0.6).

    arrow_width : float, optional
        Width of the motion arrow heads (default: 0.01).

    background_color : str, optional
        Color of the plot background (default: 'white').
    """
    import matplotlib.patches as mpatches

    prev_landmarks = prev_landmarks.landmark
    next_landmarks = next_landmarks.landmark
    x1 = [lm.x for lm in prev_landmarks]
    y1 = [lm.y for lm in prev_landmarks]
    x2 = [lm.x for lm in next_landmarks]
    y2 = [lm.y for lm in next_landmarks]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_facecolor(background_color)

    # Filter which points to display based on show_legs and show_waist

    if landmark_type == "pose":
        connections = POSE_CONNECTIONS
        
        # Filter connections based on show_legs and show_waist
        filtered_connections = []
        for start, end in connections:
            # Skip leg connections (25-32)
            if not show_legs and (start >= 25 or end >= 25):
                continue
            # Skip waist connections
            if not show_waist and ((start == 23 and end == 24) or  # hip connection
                                ((start in [11, 12] and end in [23, 24]) or  # shoulder to hip
                                 (end in [11, 12] and start in [23, 24]))):  # hip to shoulder
                continue
            filtered_connections.append((start, end))
        
        connections = filtered_connections
    else:
        connections = HAND_CONNECTIONS

        
    # Skeleton connections (prev frame) - bottom layer
    for start, end in connections:
        plt.plot([x1[start], x1[end]], [y1[start], y1[end]], 
                color=prev_connection_color,
                linestyle=prev_connection_style,
                linewidth=prev_connection_width,
                zorder=1)

    # Skeleton connections (next frame) - bottom layer
    for start, end in connections:
        plt.plot([x2[start], x2[end]], [y2[start], y2[end]], 
                color=next_connection_color,
                linestyle=next_connection_style,
                linewidth=next_connection_width,
                zorder=1)

    if landmark_type == "pose":
        visible_points = set(range(len(prev_landmarks)))
        if not show_legs:
            # Remove leg points (25-32) but keep waist points (23-24)
            for i in range(25, 33):
                visible_points.discard(i)
        if not show_waist:
            # Remove only waist points (23-24)
            visible_points.discard(23)
            visible_points.discard(24)
        
        # Only plot visible points - middle layer
        visible_x1 = [x1[i] for i in visible_points]
        visible_y1 = [y1[i] for i in visible_points]
        visible_x2 = [x2[i] for i in visible_points]
        visible_y2 = [y2[i] for i in visible_points]
        plt.scatter(visible_x1, visible_y1, c=prev_point_color, label="Current Frame", zorder=2, edgecolors=prev_connection_color)
        plt.scatter(visible_x2, visible_y2, c=next_point_color, label="Next Frame", zorder=2, edgecolors=next_connection_color)
    else:
        plt.scatter(x1, y1, c=prev_point_color, label="Current Frame", zorder=2, edgecolors=prev_connection_color)
        plt.scatter(x2, y2, c=next_point_color, label="Next Frame", zorder=2, edgecolors=next_connection_color)

    # Arrows: movement from prev to next
    for i in landmark_differences:
        if landmark_type == "pose" and i not in visible_points:
            continue

        dx = x2[i] - x1[i]
        dy = y2[i] - y1[i]
        length = np.hypot(dx, dy)
        if length > 0:
            plt.arrow(
                x1[i],
                y1[i],
                dx,
                dy,
                head_width=arrow_width,
                color=arrow_color,
                alpha=arrow_alpha,
                zorder=3,
                length_includes_head=True
            )
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.title(title, fontsize = 16)
    plt.axis("equal")

    # Create custom legend handles
    # Create a custom arrow patch for the legend
    arrow_patch = mpatches.FancyArrowPatch(
        (0.1, 0.5), (0.9, 0.5),  # Start and end points in data coordinates
        shrinkA=0, shrinkB=0,  # Don't shrink the arrow
        arrowstyle='->',  # Simple arrow style
        mutation_scale=15,  # Scale of the arrow head
        color=arrow_color,
        alpha=arrow_alpha,
        linewidth=2
    )
    
    # Create a custom handler map to properly size the arrow in the legend
    class ArrowHandler(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            # Make a copy of the arrow and scale it to fit the legend box
            patch = mpatches.FancyArrowPatch(
                (x0 + 0, y0 + height/2),  # Start point
                (x0 + width, y0 + height/2),  # End point
                shrinkA=0, shrinkB=0,
                arrowstyle='->',
                mutation_scale=15,
                color=arrow_color,
                alpha=arrow_alpha,
                linewidth=2
            )
            handlebox.add_artist(patch)
            return patch

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=prev_connection_color, markerfacecolor=prev_point_color, markersize=6, label="Current Frame", linewidth=2,linestyle=':'),
        plt.Line2D([0], [0], marker='o', color=next_connection_color, markerfacecolor=next_point_color, markersize=6, label="Next Frame", linewidth=2,linestyle=':'),
        arrow_patch  # Our custom arrow patch
    ]
    
    legend = plt.legend(handles=legend_elements, 
                       handler_map={mpatches.FancyArrowPatch: ArrowHandler()},
                       labels=["Current Frame", "Next Frame", "Difference"],
                       framealpha=1
                       )
    
    plt.tight_layout()
