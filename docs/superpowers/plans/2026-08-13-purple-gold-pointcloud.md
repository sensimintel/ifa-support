# Purple Gold Point Cloud Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `/experience` highlighter default to a black, purple-and-gold particle treatment matching the approved reference image.

**Architecture:** Keep the existing DA3 geometry and Three.js shader pipeline unchanged. Change only the centralized `_sam3hl_cfg` defaults, with a source-level regression test that verifies the complete preset remains internally consistent.

**Tech Stack:** Python, FastAPI embedded HTML/JavaScript, model-viewer, Three.js shader customization, unittest.

---

### Task 1: Lock the visual preset with a regression test

**Files:**
- Create: `tests/test_pointcloud_preset.py`
- Test: `tests/test_pointcloud_preset.py`

- [ ] **Step 1: Write the failing test**

Create a test that parses the `_sam3hl_cfg` literal from `app.py` without importing model dependencies, then asserts the approved defaults: depth color mode, dark-purple/warm-gold colors, black background, circular points, attenuation, additive blending, reduced density, restrained exposure, fog, and sparkle.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_pointcloud_preset -v`

Expected: FAIL because the current preset uses original colors, square points, normal blending, full density, and no fog or sparkle.

### Task 2: Apply the approved purple-gold defaults

**Files:**
- Modify: `app.py:1012`
- Test: `tests/test_pointcloud_preset.py`

- [ ] **Step 1: Implement the minimal preset change**

Set the centralized defaults to depth coloring, `#2a1638` to `#d8895b`, black background, 1.5 px circular attenuated points, additive blending, 82% density, reduced exposure, subtle fog, and subtle sparkle. Preserve every API field and the existing shader implementation.

- [ ] **Step 2: Run the focused test**

Run: `python -m unittest tests.test_pointcloud_preset -v`

Expected: PASS.

- [ ] **Step 3: Run syntax validation**

Run: `python -m py_compile app.py tests/test_pointcloud_preset.py`

Expected: exit code 0.

### Task 3: Run and visually verify the page

**Files:**
- Verify: `app.py`

- [ ] **Step 1: Start the existing application command**

Run the repository's documented local start command without installing or changing dependencies.

- [ ] **Step 2: Check application endpoints**

Open `/experience?demo=1` and `/sam3tune`, and confirm both render without browser console errors.

- [ ] **Step 3: Inspect the visual preset**

Confirm the point cloud uses a black background, dark-purple-to-warm-gold depth ramp, circular fine particles, restrained glow, and preserved camera interaction. Capture a screenshot for comparison when live point-cloud data is available.
