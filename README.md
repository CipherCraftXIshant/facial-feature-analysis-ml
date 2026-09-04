# Facial Feature Analysis and Age Prediction

## Project Overview
This project presents Course Evaluation 1 (CE1) for an AIML research initiative titled **"Facial Feature Analysis and Age Prediction using Machine Learning"**. The primary focus of this initial evaluation phase is establishing a rigorous data processing and feature engineering pipeline using classical machine learning methodology. 

Using the publicly available **UTKFace** dataset, we extracted structural facial landmarks, constructed scale-invariant geometric proportion features, performed exploratory data analysis (EDA), and generated a clean, model-ready dataset. No predictive models have been trained or evaluated in CE1; model building, training, and evaluation are reserved for Course Evaluation 2 (CE2).

## Problem Statement
Facial aging is a complex biological process reflected in non-linear physical changes across facial morphology, skin texture, and key landmark geometric proportions. 

The core research question investigated is: **Can measurable facial characteristics serve as reliable input features for age-related statistical analysis and future predictive modeling?**

By leveraging machine learning data preparation techniques and statistical feature extraction, we structure unorganized visual image data into standardized tabular representations. This enables quantifying linear association strength between geometric facial proportions and chronological age without making exaggerated predictive claims upfront.

## Objectives
1. **Dataset Collection & Pipeline Setup:** Acquire the UTKFace dataset, establish a local directory structure, parse image metadata from filenames, and filter corrupted or malformed filename entries.
2. **Data Cleaning & Missing Value Auditing:** Perform empirical checks for duplicate entries, invalid attribute boundaries, missing file paths on disk, and null values.
3. **Categorical Encoding:** Convert nominal categorical demographic variables (`gender` and `race`) into numerical representations using One-Hot Encoding.
4. **Facial Landmark Extraction & Feature Engineering:** Programmatically extract facial keypoints using a deep landmark detector, compute raw spatial measurements (in pixels), and derive scale-invariant geometric proportion ratios.
5. **Exploratory Data Analysis (EDA) & Selection:** Analyze feature distributions, skewness, and linear correlations with age, and construct a clean, model-ready dataset ($X, y$) for future predictive modeling.

## Dataset
- **Name:** UTKFace Dataset
- **Volume:** 23,708 raw image files downloaded locally (`.jpg` format).
- **Metadata Encoding:** Age, gender, and ethnicity attributes are encoded directly within image filenames (formatted as `[age]_[gender]_[race]_[timestamp].jpg`).

## Dataset Description
- **Metadata Parsing Results:**
  - **Successfully Parsed:** 23,705 image filenames.
  - **Excluded Malformed Filenames:** 3 filenames were excluded due to missing race fields (`39_1_20170116174525125.jpg.chip.jpg`, `61_1_20170109150557335.jpg.chip.jpg`, `61_1_20170109142408075.jpg.chip.jpg`).
- **Initial Dataset Dimensions:** 23,705 rows by 5 columns (`image_name`, `age`, `gender`, `race`, `filepath`).

## Data Cleaning
Comprehensive quality verification checks were executed on the parsed metadata:
- **Full-Row Duplicate Records:** 0 full-row duplicates found.
- **Duplicate Image Names:** 0 duplicate `image_name` entries found.
- **Value Bounds Verification:**
  - `age`: 0 rows out of range (`< 0` or `> 120`).
  - `gender`: 0 rows outside valid codes (`[0, 1]`).
  - `race`: 0 rows outside valid codes (`[0, 1, 2, 3, 4]`).
- **Disk File Path Verification:** 0 broken filepaths; all 23,705 filepaths were verified present on disk.
- **Cleaning Action:** 0 rows removed at this stage; all 23,705 records passed validation cleanly.

## Missing Value Handling
- **Before Treatment Check (`isnull()`):** 0 missing values detected across all columns (`image_name`, `age`, `gender`, `race`, `filepath`).
- **Decision Logic:** The dataset was constructed by directly parsing image filenames; filenames failing string parsing were isolated in Phase 4/5. Because no mechanism exists to introduce `NaN` values into this structured dataset, no `fillna()` or `dropna()` operations were applied.
- **After Treatment Verification:** Confirmed 0 missing values (0.0%).

## Categorical Encoding
- **Variables Encoded:** `gender` (2 categories: 0=Male, 1=Female) and `race` (5 categories: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others).
- **Encoding Method Chosen:** **One-Hot Encoding** (`pd.get_dummies`).
- **Method Rationale:** Both `gender` and `race` represent nominal (unordered) categories. Applying Label Encoding to `race` would falsely introduce synthetic ordinal relationships into statistical models (e.g., assuming Category 4 > Category 1).
- **Encoded Output:** Expanded dataset from 5 columns to 10 columns (`gender_0`, `gender_1`, `race_0`, `race_1`, `race_2`, `race_3`, `race_4`).

## Feature Extraction
- **Data Integration Note:** UTKFace does not ship with pre-existing landmark files. Facial keypoints were extracted directly from image files using OpenCV's **YuNet** face detector (`cv2.FaceDetectorYN`).
- **Execution Scope:** Performed on a documented, reproducible random sample of **3,000 images** (`random_state=42`) due to local CPU processing constraints.
- **Landmark Extraction Performance:**
  - **Successful Extractions:** 2,998 out of 3,000 images (**99.9%** success rate).
  - **Failed Extractions:** 2 out of 3,000 images (0.1%), flagged with `NaN` in measurement columns for formal handling in Phase 10.
- **Raw Spatial Measurements Computed (Pixels):**
  - `eye_distance`: Euclidean distance between left and right eye centers.
  - `mouth_width`: Euclidean distance between left and right mouth corners.
  - `face_width`: Bounding box width ($w$).
  - `face_height`: Bounding box height ($h$).
- **Exclusion of `nose_length`:** Planned `nose_length` feature was excluded because YuNet provides only one nose landmark (the nose tip), lacking a second reference point (e.g., nose bridge) required for length calculation.

## Feature Engineering
To eliminate scale dependency caused by varying camera distances and crops, 3 scale-invariant ratio features were derived using vectorized operations:

1. **`face_aspect_ratio`** = $\frac{\text{face\_height}}{\text{face\_width}}$
   - *Rationale:* Measures facial morphology (elongated vs. round face shape) independently of image zoom or bounding box resolution.
2. **`eye_to_face_ratio`** = $\frac{\text{eye\_distance}}{\text{face\_width}}$
   - *Rationale:* Normalizes inter-ocular distance relative to facial width across subject images.
3. **`mouth_to_face_ratio`** = $\frac{\text{mouth\_width}}{\text{face\_width}}$
   - *Rationale:* Normalizes mouth width relative to facial width to ensure scale-invariance.

*Note:* Rows with `NaN` measurement values from failed extractions (2 rows) were dropped prior to ratio computation, leaving **2,998 clean rows**.

## Feature Selection
- **Feature Matrix ($X$):** 14 predictor variables:
  - Encoded categoricals: `gender_0`, `gender_1`, `race_0`, `race_1`, `race_2`, `race_3`, `race_4`
  - Raw measurements: `eye_distance`, `mouth_width`, `face_width`, `face_height`
  - Engineered ratios: `face_aspect_ratio`, `eye_to_face_ratio`, `mouth_to_face_ratio`
- **Target Vector ($y$):** `age`
- **Excluded Columns:** `image_name` and `filepath` (retained in intermediate files as metadata identifiers, excluded from $X$).
- **Linear Association Strength (Pearson $r$ with Age, sorted by $|r|$):**
  - `face_aspect_ratio`: $+0.254$
  - `mouth_to_face_ratio`: $+0.241$
  - `eye_to_face_ratio`: $+0.212$
  - `face_width`: $-0.186$
  - `face_height`: $+0.168$
  - `mouth_width`: $+0.111$
  - `eye_distance`: $-0.011$
  - *Finding:* All linear correlations with age are weak ($|r| < 0.30$).

## Exploratory Data Analysis
Five visualization figures were generated and saved to `visualizations/`:
1. **Age Distribution (`01_age_distribution.png`):** Right-skewed distribution. Mean age = 33.5 years, Median = 30.0 years, Range = 1 to 115 years.
2. **Gender Distribution (`02_gender_distribution.png`):** Fairly balanced distribution (Male: 1,576 / 52.6%, Female: 1,422 / 47.4%).
3. **Race Distribution (`03_race_distribution.png`):** Imbalanced distribution (`Race_0`: 1,307 / 43.6%, `Race_1`: 541 / 18.0%, `Race_3`: 502 / 16.7%, `Race_2`: 423 / 14.1%, `Race_4`: 225 / 7.5%).
4. **Age vs. Top Features Scatter Plots (`04_age_vs_features_scatter.png`):** 2x2 grid displaying Age vs `face_aspect_ratio`, `mouth_to_face_ratio`, `eye_to_face_ratio`, and `face_width`. Visual inspection confirms weak, noisy linear associations consistent with correlation coefficients.
5. **Feature Distributions & Skewness (`05_feature_distributions.png`):** 2x3 grid displaying numeric feature histograms. Engineered ratios exhibit near-symmetric distributions (`face_aspect_ratio` skew = -0.010, `eye_to_face_ratio` skew = +0.120). Raw pixel measurements exhibit mild-to-moderate skewness (`face_height` most skewed at -0.786).

## Final Preprocessed Dataset
- **Dimensions:** 2,998 rows by 15 columns (`age` target + 14 predictor features).
- **Missing Values:** 0 missing values (100% complete).
- **Saved Output Path:** `data/processed/utkface_final_preprocessed.csv`

## CE1 Conclusion
Course Evaluation 1 successfully established an end-to-end data acquisition, cleaning, landmark extraction, feature engineering, and EDA pipeline. Empirical findings confirm that linear correlations between individual 2D geometric facial ratios and chronological age are weak ($|r| \le 0.254$). No age-prediction models were trained in CE1.

## Future Scope
In Course Evaluation 2 (CE2), the project will advance to predictive machine learning modeling:
- **Train/Test Splitting:** Partition `utkface_final_preprocessed.csv` into training and validation sets.
- **Feature Scaling:** Apply standardization/normalization if required by specific estimators.
- **Baseline Regression Models:** Implement Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression.
- **Model Evaluation:** Evaluate model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
- **Non-Linear Exploration:** Explore non-linear algorithms or multi-feature interactions to model complex age-morphology relationships given the weak linear associations observed in CE1.

## Project Structure
```
facial-feature-analysis-ml/
├── data/
│   ├── raw/                           # Raw UTKFace image files (git-ignored)
│   └── processed/                     # Intermediate & final preprocessed datasets
│       ├── utkface_metadata.csv
│       ├── utkface_cleaned.csv
│       ├── utkface_missing_handled.csv
│       ├── utkface_encoded.csv
│       ├── utkface_with_measurements.csv
│       ├── utkface_engineered.csv
│       ├── X_features.csv
│       ├── y_target.csv
│       └── utkface_final_preprocessed.csv
├── notebooks/
│   ├── 01_setup_and_environment.ipynb
│   ├── 02_dataset_collection_and_description.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_missing_value_handling.ipynb
│   ├── 05_categorical_encoding.ipynb
│   ├── 06_landmark_extraction_test.ipynb
│   ├── 07_feature_extraction.ipynb
│   ├── 08_feature_engineering.ipynb
│   ├── 09_feature_selection.ipynb
│   ├── 10_eda_and_visualization.ipynb
│   └── 11_final_preprocessing.ipynb
├── visualizations/                    # Saved EDA figures (.png)
│   ├── 01_age_distribution.png
│   ├── 02_gender_distribution.png
│   ├── 03_race_distribution.png
│   ├── 04_age_vs_features_scatter.png
│   └── 05_feature_distributions.png
├── README.md                          # Project documentation
├── requirements.txt                   # Dependency specifications
└── .gitignore                         # Version control exclusions
```

## Technologies Used
- **Python 3.13**
- **NumPy:** Numerical computations and vector operations.
- **Pandas:** Data structures, tabular data manipulation, and CSV I/O.
- **Matplotlib:** Data distribution histograms, bar charts, and scatter plot visualizations.
- **OpenCV (`cv2.FaceDetectorYN` / YuNet):** Deep learning-based face detection and facial landmark extraction.
- **Scikit-Learn:** Categorical pre-processing (`OneHotEncoder`).
