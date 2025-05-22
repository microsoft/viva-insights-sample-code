# Calculated Columns for Copilot Usage Segmentation

This folder contains DAX scripts for calculated columns used to segment users by their Copilot usage patterns in Power BI.

## RL12W (12-week rolling) Approach

**Recommended if the adoption timeframe is longer than 12 weeks, and the goal is to evaluate long term habit formation on Copilot usage.**

- **_Total Copilot actions_RL12W**:  
  Calculates the average weekly Copilot actions for each user over the last 12 weeks.

- **_IsHabit_RL12W**:  
  Flags users as "habitual" if they used Copilot in at least 9 of the last 12 weeks.

- **_CopilotUsageSegment_RL12W**:  
  Categorizes users as Power, Habitual, Novice, Low, or Non-users based on their calculated usage and habit status.

### Definitions

- Power User: average 15+ weekly total Copilot actions and any use of Copilot in 9 of the last 12 weeks.
- Habitual User: any use of Copilot in 9 of the last 12 weeks.
- Novice User: average 1+ weekly Copilot actions over the past 12 weeks.
- Low User: at least one Copilot action in the past 12 weeks.
- Non−user: zero Copilot actions in the past 12 weeks.

## RL4W (4-week rolling) Approach

**Recommended if the adoption period is shorter than 12 weeks, or adoption is being evaluated as part of an agile pilot program.**

- **_Total Copilot actions_RL4W**:  
  Calculates the average weekly Copilot actions for each user over the last 4 weeks.

- **_IsHabit_RL4W**:  
  Flags users as "habitual" if they used Copilot in all of the last 4 weeks.

- **_CopilotUsageSegment_RL4W**:  
  Categorizes users as Power, Habitual, Novice, Low, or Non-users based on their calculated usage and habit status.

### Definitions

- Power User: average 15+ weekly total Copilot actions and any use of Copilot in all of the last 4 weeks.
- Habitual User: any use of Copilot in all of the last 4 weeks.
- Novice User: average 1+ weekly Copilot actions over the past 4 weeks.
- Low User: at least one Copilot action in the past 4 weeks.
- Non−user: zero Copilot actions in the past 4 weeks.

## Usage

1. Open Power BI Desktop and select your table in Data view.
2. For each calculated column you want to add, open the corresponding `.dax` file in this folder.
3. Copy the DAX expression and paste it into Power BI as a 'New Column' formula.
4. Use the resulting columns to analyze and visualize Copilot adoption and habit formation.

See the main tutorial for more details on usage and definitions.
