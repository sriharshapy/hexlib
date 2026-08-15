# hexlib/device/qdc/__init__.py
"""Qualcomm Device Cloud plumbing: stage an artifact, submit it, and detect
completion without ever trusting job status or the jobs list.

qualcomm_device_cloud_sdk is imported lazily, inside functions, in job.py --
never at module import time -- so `import hexlib.device.qdc` and the offline
tests in hexlib/tests/test_qdc.py work on a machine where that package is
not installed. It is only required once code actually talks to the network.
"""
