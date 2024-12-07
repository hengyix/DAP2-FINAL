# Global Terrorism Analysis (2000-2020)
This is the final project for DAP2 course, done by Sienna Wang and Hengyi Xing.  
This project analyzes global terrorism data from 2000 to 2020. The detailed research findings can be found in **`write-up.pdf`**. Below is an overview of the project structure and instructions for reproducing our results.

## Project Overview

### Dynamic Plots
- A **Global Terrorism Heatmap** and a **Regional Active Groups Bar Chart** are included in the shiny app.
  - Users can interact with the visualizations by selecting metrics, years, or regions of interest.
  - Detailed information is available by hovering over the plots.

### Static Plots
- Two **correlation plots**:
  - `pictures/democracy.png`
  - `pictures/economy.png`
- Two **NLP bar charts**:
  - `pictures/motive_taliban.png`
  - `pictures/motive_ISIL.png`

## Reproducibility Instructions

### Step 1: Clone the Repository
Clone the project repository to your local system.

### Step 2: Install Required Packages
Install the required Python packages listed in **`requirements.txt`**:
```bash
pip install -r requirements.txt
```

### Step 3: Download Additional Data
1. Download the additional dataset from this [link](https://drive.google.com/file/d/1L_0mg8PEYIpWt4vC2UssMU8i8u1Vhnlu/view?usp=sharing).
2. Save the file in the `data` folder and the `app-py` folder.

### Step 4: Reproduce Static Plots
1. Open and execute **`write-up.qmd`**.
2. This will:
- Generate static plots saved in the `pictures` folder.
- Generate an HTML file that provides a walkthrough of the entire project, displaying both the static plots created and screenshots of the dynamic plots.

### Step 5: Run the Dynamic Plot
1. Navigate to the `app-py` folder.
2. Launch the shiny app:
   ```bash
   shiny run app.py
   ```
3. Explore the shiny app:
- Page **"Global Overview"**: View the dynamic heatmap.
- Page **"Regional Group Analysis"**: Explore the regional active groups bar chart.


## Minimal Setup for Shiny App
If you're only interested in the shiny app:
- Download the `shiny-app` folder along with the required packages and data.

---

Thank you for exploring our project!
