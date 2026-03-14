import pandas as pd

data = {
    "complaint_text": [
        "Water leak in Trilokpuri block 14",
        "Street light not working in Dwarka sector 10",
        "Garbage not collected near Mayur Vihar phase 2",
        "Road pothole in Okhla phase 2",
        "Broken footpath in Rohini sector 3",
        "Overflowing drains in Narela Mandi",
        "Broken street lamp in Shastri Nagar",
        "Illegal dumping of waste in Trilokpuri block 18",
        "Park lights not working in Dwarka sector 5",
        "Waterlogging near Mayur Vihar phase 1",
        "Potholes near Okhla phase 3",
        "Missing manhole cover in Rohini sector 7",
        "Street cleaning required in Narela Ind. Area",
        "Tree roots damaging footpath in Trilokpuri block 2",
        "Overflowing garbage bins in Mayur Vihar phase 2",
        "Broken street sign in Dwarka sector 22",
        "Road repair needed in Okhla phase 2",
        "Streetlight flickering in Rohini sector 4",
        "Blocked drains in Shastri Nagar",
        "Garbage collection delayed in Narela Mandi"
    ]
}
df = pd.DataFrame(data)
df.to_csv("complaints.csv", index=False)
print("complaints.csv created!")