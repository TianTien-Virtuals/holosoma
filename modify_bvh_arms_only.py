#!/usr/bin/env python3
"""
Modify BVH file to keep only arm movements, freeze legs/hips at initial pose.

This modifies the MOTION data (not the hierarchy) so that:
- Arms move as in original motion
- Legs, hips stay at their Frame 0 position (static)
- Result: Human with dancing arms but standing still
"""

import numpy as np
import sys
from pathlib import Path

def modify_bvh_freeze_legs(input_bvh, output_bvh):
    """
    Modify BVH file to freeze leg movements while keeping arm movements.
    
    Args:
        input_bvh: Path to original BVH file
        output_bvh: Path to save modified BVH file
    """
    with open(input_bvh, 'r') as f:
        lines = f.readlines()
    
    # Find MOTION section
    motion_start = -1
    for i, line in enumerate(lines):
        if line.strip() == 'MOTION':
            motion_start = i
            break
    
    if motion_start == -1:
        raise ValueError("No MOTION section found")
    
    # Parse hierarchy to understand channel structure
    print("Parsing BVH hierarchy...")
    hierarchy_lines = lines[:motion_start]
    
    # Extract channel info for each joint
    joint_channels = []  # List of (joint_name, num_channels, start_index)
    channel_offset = 0
    current_joint = None
    
    for line in hierarchy_lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith('ROOT') or line_stripped.startswith('JOINT'):
            parts = line_stripped.split()
            current_joint = parts[1] if len(parts) > 1 else "Unknown"
            
        elif line_stripped.startswith('CHANNELS'):
            parts = line_stripped.split()
            num_channels = int(parts[1])
            joint_channels.append((current_joint, num_channels, channel_offset))
            channel_offset += num_channels
    
    print(f"\nFound {len(joint_channels)} joints with channels:")
    for joint, num_ch, offset in joint_channels:
        print(f"  {joint:20s}: {num_ch} channels starting at index {offset}")
    
    # Parse motion section
    motion_lines = lines[motion_start+1:]
    frames_line = motion_lines[0].strip()
    frame_time_line = motion_lines[1].strip()
    
    num_frames = int(frames_line.split(':')[1].strip())
    frame_time = float(frame_time_line.split(':')[1].strip())
    
    print(f"\nMotion info:")
    print(f"  Frames: {num_frames}")
    print(f"  Frame time: {frame_time}s ({1/frame_time:.1f} FPS)")
    print(f"  Total channels: {channel_offset}")
    
    # Parse motion data
    motion_data = []
    for line in motion_lines[2:]:
        line = line.strip()
        if line:
            values = [float(x) for x in line.split()]
            motion_data.append(values)
    
    motion_data = np.array(motion_data)
    print(f"  Motion data shape: {motion_data.shape}")
    
    # Identify which channels to freeze
    # Keep ONLY: LeftShoulder, LeftArm, LeftForeArm, LeftHand,
    #            RightShoulder, RightArm, RightForeArm, RightHand
    # Freeze: Everything else (hips, legs, spine, neck, head)
    
    # Define joints that should MOVE (arms only)
    keep_moving = ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                   'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
    
    channels_to_freeze = []
    for joint, num_ch, offset in joint_channels:
        # If joint is NOT in the keep_moving list, freeze it
        if joint not in keep_moving:
            # Freeze ALL channels for this joint
            for ch_idx in range(num_ch):
                channels_to_freeze.append(offset + ch_idx)
    
    print(f"\nFreezing {len(channels_to_freeze)} channels:")
    print(f"  Channel indices: {channels_to_freeze[:10]}... (showing first 10)")
    
    # Create modified motion data
    modified_motion = motion_data.copy()
    
    # Get Frame 0 values for channels to freeze
    frame0_values = motion_data[0, channels_to_freeze]
    
    # Set all frames to Frame 0 values for frozen channels
    for frame_idx in range(num_frames):
        modified_motion[frame_idx, channels_to_freeze] = frame0_values
    
    # Verify
    variance_before = np.var(motion_data[:, channels_to_freeze], axis=0).mean()
    variance_after = np.var(modified_motion[:, channels_to_freeze], axis=0).mean()
    print(f"\nVariance check (should be ~0 after):")
    print(f"  Before: {variance_before:.6f}")
    print(f"  After: {variance_after:.6f}")
    
    # Write modified BVH
    output_lines = lines[:motion_start+3].copy()  # Header + MOTION + Frames + Frame Time
    
    # Add modified motion data
    for frame in modified_motion:
        frame_str = ' '.join(f'{v:.6f}' for v in frame)
        output_lines.append(frame_str + '\n')
    
    with open(output_bvh, 'w') as f:
        f.writelines(output_lines)
    
    print(f"\n✅ Modified BVH saved to: {output_bvh}")
    print(f"\nDifference from original:")
    print(f"  - FROZEN: Hips (all 6 DOFs - no position or rotation)")
    print(f"  - FROZEN: All leg joints (no leg movement)")
    print(f"  - FROZEN: Spine, Spine1, Spine2 (no torso bend)")
    print(f"  - FROZEN: Neck, Head (no head movement)")
    print(f"  - MOVING: LeftShoulder→LeftHand, RightShoulder→RightHand only!")
    print(f"\nTotal: {69 - len(channels_to_freeze)} moving channels (arms only)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_bvh_arms_only.py input.bvh [output.bvh]")
        print("\nExample:")
        print("  python modify_bvh_arms_only.py dance2_subject5.bvh dance2_subject5_arms_only.bvh")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.bvh', '_arms_only.bvh')
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("="*60)
    
    modify_bvh_freeze_legs(input_file, output_file)

