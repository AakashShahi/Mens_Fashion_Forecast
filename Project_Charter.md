# Project Charter: Kathmandu Youth Fashion Forecaster

## 1. Project Initiation
**Mission:** To revolutionize inventory planning for Nepali fashion retailers by legally predicting male youth fashion trends using AI and behavioral psychology.

**Primary Business Question:**
"What specific clothing items will high-fashion male youth (17-25) in Kathmandu want to buy in the next 30 days?"

**Success Metrics:**
1.  **Forecast Accuracy:** RMSE < 15 units for high-volume items.
2.  **Stockout Reduction:** Reduce missed sales opportunities by 20%.
3.  **Overstock Reduction:** Reduce unsold inventory by 15%.
4.  **User Understanding:** Dashboard insights must be understandable by non-technical store managers (measured by "Recommendation Clarity" score).

## 2. Data Sources & Methodology
| Data Source | Type | Purpose | Collection Frequency |
| :--- | :--- | :--- | :--- |
| **Sales Records** | Internal SQL/CSV | Historical baseline & seasonality | Daily |
| **Instagram/TikTok** | Text/Hashtags | Detecting "Buzz" & aesthetic tribes | Weekly |
| **Google Trends** | Search Volume Interest | Macroscopic demand shifts | Weekly |
| **Fashion Psychology** | Domain Rules | Weighting trends by cultural adoptability | Static (Review Monthly) |

## 3. Scope & Limitations
- **Scope:** Male clothing, ages 17-25, Kathmandu valley.
- **Limitations:**
    - Social media data is proxy-based (hashtags), not visual recognition.
    - Psychology scores are heuristic-based, not individual psychometrics.
    - External shocks (e.g., fuel crisis, sudden lockdowns) are not modeled.

## 4. Stakeholders
- **Primary User:** Inventory Managers / Shop Owners.
- **Secondary User:** Marketing Team (for ad targeting).
