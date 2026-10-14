import numpy as np
import time
import pathlib
import os
import json
import h5py
from multiprocessing.pool import ThreadPool
from os.path import join, dirname

from ridge_utils.npp import zscore, mcorr
from ridge_utils.utils import make_delayed
from config import DATA_DIR, EM_DATA_DIR

def apply_zscore_and_hrf(stories, downsampled_feat, trim, ndelays):
    """Get (z-scored and delayed) stimulus for train and test stories.
    The stimulus matrix is delayed (typically by 2,4,6,8 secs) to estimate the
    hemodynamic response function with a Finite Impulse Response model.

    Args:
        stories: List of stimuli stories.

    Variables:
        downsampled_feat (dict): Downsampled feature vectors for all stories.
        trim: Trim downsampled stimulus matrix.
        delays: List of delays for Finite Impulse Response (FIR) model.

    Returns:
        delstim: <float32>[TRs, features * ndelays]
    """
    stim = [zscore(downsampled_feat[s][5+trim:-trim]) for s in stories]
    stim = np.vstack(stim)
    delays = range(1, ndelays+1)
    delstim = make_delayed(stim, delays)
    return delstim

def get_response(stories, subject):
    """Get the subject"s fMRI response for stories."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = join(DATA_DIR, "derivatives/preprocessed_data/%s" % subject)
    base = os.path.join(main_path, subject_dir)
    resp = []
    valid_stories = []
    for story in stories:
        resp_path = os.path.join(base, "%s.hf5" % story)
        if not os.path.exists(resp_path):
            print(f"WARNING: Response file not found for story '{story}': {resp_path}")
            continue
        
        try:
            hf = h5py.File(resp_path, "r")
            resp.extend(hf["data"][:])
            hf.close()
            valid_stories.append(story)
        except (OSError, KeyError) as e:
            print(f"WARNING: Could not load response for story '{story}': {type(e).__name__}")
            print(f"  Details: {e}")
            continue
    
    if not resp:
        raise ValueError(
            f"No valid response files found for subject {subject} "
            f"in stories {stories}"
        )
    
    return np.array(resp), valid_stories

def get_story_split(sessions):
    """
    Return ordered training and test story lists for the selected sessions.
    """
    with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)

    train_stories = []
    test_stories = []

    for session in sessions:
        session = str(session)

        if session not in sess_to_story:
            raise ValueError(f"Unknown session: {session}")

        stories, test_story = sess_to_story[session]

        train_stories.extend(stories)

        if test_story not in test_stories:
            test_stories.append(test_story)

    # Remove duplicates while preserving order
    train_stories = list(dict.fromkeys(train_stories))
    test_stories = list(dict.fromkeys(test_stories))

    overlap = set(train_stories).intersection(test_stories)
    if overlap:
        raise ValueError(f"Training and test stories overlap: {sorted(overlap)}")

    return train_stories, test_stories


def get_permuted_corrs(true, pred, blocklen):
	nblocks = int(true.shape[0] / blocklen)
	true = true[:blocklen*nblocks]
	block_index = np.random.choice(range(nblocks), nblocks)
	index = []
	for i in block_index:
		start, end = i*blocklen, (i+1)*blocklen
		index.extend(range(start, end))
	pred_perm = pred[index]
	nvox = true.shape[1]
	corrs = np.nan_to_num(mcorr(true, pred_perm))
	return corrs

def permutation_test(true, pred, blocklen, nperms):
	start_time = time.time()
	pool = ThreadPool(processes=10)
	perm_rsqs = pool.map(
		lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
	pool.close()
	end_time = time.time()
	print((end_time - start_time) / 60)
	perm_rsqs = np.array(perm_rsqs).astype(np.float32)
	real_rsqs = np.nan_to_num(mcorr(true, pred))
	pvals = (real_rsqs <= perm_rsqs).mean(0)
	return np.array(pvals), perm_rsqs, real_rsqs

def run_permutation_test(zPresp, pred, blocklen, nperms, mode='', thres=0.001):
	assert zPresp.shape == pred.shape, print(zPresp.shape, pred.shape)

	start_time = time.time()
	ntr, nvox = zPresp.shape
	partlen = nvox
	pvals, perm_rsqs, real_rsqs = [[] for _ in range(3)]

	for start in range(0, nvox, partlen):
		print(start, start+partlen)
		pv, pr, rs = permutation_test(zPresp[:, start:start+partlen], pred[:, start:start+partlen],
									  blocklen, nperms)
		pvals.append(pv)
		perm_rsqs.append(pr)
		real_rsqs.append(rs)
	pvals, perm_rsqs, real_rsqs = np.hstack(pvals), np.hstack(perm_rsqs), np.hstack(real_rsqs)

	assert pvals.shape[0] == nvox, (pvals.shape[0], nvox)
	assert perm_rsqs.shape[0] == nperms, (perm_rsqs.shape[0], nperms)
	assert perm_rsqs.shape[1] == nvox, (perm_rsqs.shape[1], nvox)
	assert real_rsqs.shape[0] == nvox, (real_rsqs.shape[0], nvox)

	cci.upload_raw_array(os.path.join(save_location, '%spvals'%mode), pvals)
	cci.upload_raw_array(os.path.join(save_location, '%sperm_rsqs'%mode), perm_rsqs)
	cci.upload_raw_array(os.path.join(save_location, '%sreal_rsqs'%mode), real_rsqs)
	print((time.time() - start_time)/60)
	
	pID, pN = fdr_correct(pvals, thres)
	cci.upload_raw_array(os.path.join(save_location, '%sgood_voxels'%mode), (pvals <= pN))
	cci.upload_raw_array(os.path.join(save_location, '%spN_thres'%mode), np.array([pN, thres], dtype=np.float32))
	return
