# Candidate open EEG datasets (beginner goal)

**Goal:** find open EEG datasets where the same untrained-CNN analysis could be run.
Good candidates have resting or simple-task EEG, a binary participant label
(like patient vs control), and BIDS formatting.

Starter list. Please **check each entry on its OpenNeuro / website page**, fill in
the empty cells, and add more. Mark the "Checked" column with your GitHub handle once verified.

| Dataset | What | Participants | Channels | Label for a probe | Format | Access | Checked |
|---|---|---|---|---|---|---|---|
| [ds003478](https://openneuro.org/datasets/ds003478) | Resting EEG, eyes open/closed | 122 | 64 | High vs low BDI (depression) | BIDS, EEGLAB | Open | ✅ (used here) |
| [ds004504](https://openneuro.org/datasets/ds004504) | Resting EEG, eyes closed | 88 (36 AD, 23 FTD, 29 C) | 19 | Alzheimer's / FTD / control | BIDS | Open | ✅ (used) |
| [ds002778](https://openneuro.org/datasets/ds002778) | Resting EEG | 31 (15 PD off medication, 16 C) | 32 | Parkinson's vs control | BIDS | Open | ✅ (used) |
| [ds003775](https://openneuro.org/datasets/ds003775) | Resting EEG (SRM) | ~111 | 64 | Healthy only (age/sex?) | BIDS | Open | not used |
| [MPI-LEMON](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) | Resting EEG, eyes open/closed | 203 with EEG (137 young, 65 old) | 61 | Young vs old | BrainVision; authors' preprocessed EEGLAB | Open | ✅ (used) |
| [TUH EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) | Clinical EEG | thousands | varies | Normal vs abnormal (TUAB) | EDF | Free registration | |

Also used: BNCI2014_001 (BCI Competition IV 2a, 9 subjects, left vs right hand imagery) and
Sleep-EDF Expanded sleep cassette (first 20 subjects, 5 sleep stages). Configs for all six are in
`configs/`; each is downloaded on first use.

## What to record for each dataset

- Number of participants and how many in each label group
- Sampling rate, number of channels, recording length
- Eyes open / eyes closed markers? Task?
- License and how to download (S3, DataLad, website)
- Known quirks from the dataset README

## Searching for more

- [OpenNeuro](https://openneuro.org): filter by modality = EEG
- [NEMAR](https://nemar.org): OpenNeuro EEG data with ready-made visualizations
- Papers that benchmark EEG deep learning often list their datasets
