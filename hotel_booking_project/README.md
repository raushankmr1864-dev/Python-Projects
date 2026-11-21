# Project 13: Hotel Booking & Occupancy Analysis

Uses only **pandas** and **numpy** (plus Python's built-in `os` for file paths). No SQL, no other libraries.

## How to run
1. Install Python 3 and the two libraries: `pip install pandas numpy`
2. Unzip this folder, open a terminal inside it, and run:
   `python hotel_analysis.py`
3. Results print on screen and are saved in the `output/` folder.

## Folder contents
| File / folder | Purpose |
|---|---|
| `hotel_analysis.py` | Complete source code (Tasks 1-15, commented) |
| `data/13_hotel_bookings.csv` | Dataset (database) |
| `output/analysis_report.txt` | Full printed report with all task results |
| `output/room_type_summary.csv` | Deliverable: room-type revenue summary |
| `output/channel_cancellation_table.csv` | Deliverable: channel cancellation table |
| `output/monthly_occupancy_trend.csv` | Deliverable: monthly occupancy trend |
| `output/` (other CSVs) | Channel ranking, lead-time cancellation, weekday arrivals, traveler summary, cleaned data, etc. |

## Task-to-code map
| Task | Section in `hotel_analysis.py` |
|---|---|
| 1 Load & inspect | TASK 1 |
| 2 Missing values & duplicate IDs | TASK 2 |
| 3 Handle missing values | TASK 3 (median rate per room type, overall median lead time) |
| 4 Datetime, month, weekday | TASK 4 |
| 5 Bookings, nights, cancellation % | TASK 5 |
| 6 Revenue = Nights x Room_Rate | TASK 6 (canceled = 0) |
| 7 Room-type rate & occupancy | TASK 7 (`groupby(['Room_Type','Booking_Channel'])` included) |
| 8 Rank channels | TASK 8 |
| 9 Cancellation by channel & lead band | TASK 9 |
| 10 Top 3 revenue months | TASK 10 |
| 11 Leisure vs business stay | TASK 11 |
| 12 NumPy mean/median/std | TASK 12 |
| 13 IQR outliers (`np.percentile`) | TASK 13 |
| 14 `pd.cut()` lead-time bands | TASK 14 |
| 15 Five recommendations | TASK 15 (built from the computed results) |

## Key results (from the supplied dataset)
- 761 rows loaded; 11 exact duplicate Booking_IDs removed -> **750 bookings**
- 18 missing Lead_Time_Days and 18 missing Room_Rate values filled with medians
- **2,988 room nights** booked; **19.87% cancellation**; 2,393 room nights actually stayed
- Realized revenue: **21,080,871.94**
- Standard rooms give the most revenue (50.6%); Suites have the highest average rate (9,238.78)
- OTA is #1 by revenue, Corporate is #1 by volume
- Cancellation is highest on Direct Website (23.03%) and lowest on OTA (14.92%)
- Top 3 revenue months: 2025-10, 2026-04, 2025-05
- Business vs Leisure average stay: 4.02 vs 3.94 nights (almost no difference)
- Daily room rate (NumPy): mean 8,759.70, median 8,864.78, std 3,606.76
- IQR method finds 1 unusually high booking (B00004, 104,291.32)

## Assumptions
- **Occupancy**: the data has no hotel room capacity, so true occupancy % cannot be computed. Occupancy here means room nights actually stayed (non-canceled) and each room type's or month's share of them.
- **Duplicates**: all 11 repeated Booking_IDs were identical rows, so the first copy was kept.
- **Lead-time bands** (`pd.cut`): Last-Minute 0-7 days, Short-Term 8-30, Medium-Term 31-90, Advance 91+.
- **Revenue** is counted only for non-canceled bookings.
