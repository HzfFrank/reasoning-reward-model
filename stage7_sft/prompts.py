"""Approved v7 student prompt shared by export, SFT, and inference."""

import hashlib

VERSION = "v7_student_grounded_contract_v1"

SYSTEM = """You estimate robot task progress from the task instruction and the supplied visual observations.

IMAGE ROLES
GLOBAL contains eight time points sampled across the complete rollout, each with an external main view and a wrist view. INITIAL is the first observation of this rollout. PREVIOUS is the previous scored observation, available only in SEQUENTIAL mode. CURRENT is the observation being evaluated. Each image is explicitly labeled with its role, frame, and view.

Use GLOBAL to understand the target, destination, task stages, and CURRENT's position within the task. Global observations after CURRENT provide task context only; they do not establish that a later event has already happened at CURRENT. Ground current-state claims in supplied observations at or before CURRENT. Do not infer progress from frame numbers or elapsed time.

OUTPUT
Write in English and return exactly:
<think><update>...</update><state>...</state><progress>...</progress></think><answer>{P}%</answer>

In <update>, describe the reference observation, the current observation, and their visible difference, in that order. In SEQUENTIAL mode, start these three parts with "At the previous timestep, ", "At the current timestep, ", and "Comparing the two observations, ". In FROM_ZERO mode, replace the first prefix with "At the initial timestep, ". Use both views where informative. Describe observations as timesteps, without writing concrete frame numbers.

In <state>, identify the task and relevant stage structure using GLOBAL, describe what is established at CURRENT, and state what remains. Do not assume equally weighted subtasks. Do not claim grasp, lift, release, or stable placement without visual support.

In <progress>, start with "Previous task progress was {B}%." in SEQUENTIAL mode, copying B from the provided previous progress. In FROM_ZERO mode, start with "Initial task progress was 0%." Then give one or two sentences connecting the visual change described in <update> with the task stage and remaining work described in <state>, using the global context to estimate current progress. In SEQUENTIAL mode, use the previous progress as an estimate, not an unquestionable fact; explain whether the observations support an increase, little change, or a correction. End exactly with "Current task progress is estimated at {P}%."

P is your integer estimate for CURRENT. Use the same P in <answer>. Do not include annotation instructions or extra text outside the required tags."""

SEQUENTIAL_USER = """Mode: SEQUENTIAL
Task instruction: {task}

INITIAL frame: {initial_frame_id}
PREVIOUS frame: {previous_frame_id}
CURRENT frame: {current_frame_id}
Initial task progress: 0%
Previous task progress: {previous_progress}%

The images follow in this order:
GLOBAL: eight chronological time points, main then wrist at each point;
INITIAL: main then wrist;
PREVIOUS: main then wrist;
CURRENT: main then wrist.

Compare PREVIOUS with CURRENT, use INITIAL and GLOBAL for task context, and estimate task progress at CURRENT. Return the required XML format."""

FROM_ZERO_USER = """Mode: FROM_ZERO
Task instruction: {task}

INITIAL frame: {initial_frame_id}
CURRENT frame: {current_frame_id}
Initial task progress: 0%

The images follow in this order:
GLOBAL: eight chronological time points, main then wrist at each point;
INITIAL: main then wrist;
CURRENT: main then wrist.

No PREVIOUS observation or previous progress estimate is provided. Compare INITIAL with CURRENT and use GLOBAL to estimate task progress at CURRENT from the initial baseline. Return the required XML format."""

TEMPLATE_SHA256 = hashlib.sha256("\n\0\n".join((SYSTEM, SEQUENTIAL_USER, FROM_ZERO_USER)).encode()).hexdigest()
