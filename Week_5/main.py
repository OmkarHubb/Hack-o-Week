import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


players = pd.read_csv("fifa_players.csv")
clubs = pd.read_csv("club_info.csv")


players.fillna(0, inplace=True)


class Player:
    def __init__(self, name, club, overall):
        self.name = name
        self.club = club
        self.overall = overall

    def display(self):
        print("\nSample Player Object")
        print("--------------------")
        print("Name :", self.name)
        print("Club :", self.club)
        print("Overall :", self.overall)

sample = Player("Lionel Messi", "Inter Miami", 90)



def show_players():
    print("\nAll FIFA Players")
    print(players)


def top_players():
    top = players.sort_values(by="Overall", ascending=False)
    print("\nTop 5 Players")
    print(top.head())


def average_rating():

    ratings = np.array(players["Overall"])

    print("\nNumPy Array")
    print(ratings)

    print("\nAverage Rating :", np.mean(ratings))
    print("Highest Rating :", np.max(ratings))
    print("Lowest Rating :", np.min(ratings))


def best_clubs():

    print("\nAverage Rating by Club")

    club_avg = players.groupby("Club")["Overall"].mean()

    print(club_avg)


def merge_data():

    merged = pd.merge(players, clubs, on="Club")

    print("\nMerged Data")

    print(merged)


def player_names():

    names = [name for name in players["Name"]]

    print("\nPlayer Names (List Comprehension)\n")

    print(names)


def show_object():

    sample.display()


def show_bar_graph():

    top = players.sort_values(by="Overall", ascending=False)

    plt.figure(figsize=(8,5))

    plt.bar(top["Name"], top["Overall"])

    plt.xticks(rotation=45)

    plt.title("Top FIFA Player Ratings")

    plt.xlabel("Players")

    plt.ylabel("Overall")

    plt.tight_layout()

    plt.show()


def show_scatter():

    sns.scatterplot(data=players, x="Age", y="Overall")

    plt.title("Age vs Overall")

    plt.show()
   

while True:

    print("\n===================================")
    print("      FIFA PLAYER ANALYSIS")
    print("===================================")
    print("1. Show All Players")
    print("2. Top 5 Players")
    print("3. Average Overall Rating")
    print("4. Best Clubs")
    print("5. Merge Club Information")
    print("6. Show Sample Player (OOP)")
    print("7. Show Player Names")
    print("8. Bar Graph")
    print("9. Scatter Graph")
    print("10. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        show_players()

    elif choice == "2":
        top_players()

    elif choice == "3":
        average_rating()

    elif choice == "4":
        best_clubs()

    elif choice == "5":
        merge_data()

    elif choice == "6":
        show_object()

    elif choice == "7":
        player_names()

    elif choice == "8":
        show_bar_graph()

    elif choice == "9":
        show_scatter()

    elif choice == "10":
        print("\nThank You for using FIFA Player Analysis!")
        break

    else:
        print("\nInvalid Choice! Please try again.")