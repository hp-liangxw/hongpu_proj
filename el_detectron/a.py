

answer = input("Is the configuration file correct? [yes/no]:")
while answer.upper() not in ["YES", "Y", "NO", "N"]:
    print("Please input yes or no!")
    answer = input()

if answer.upper() in ["YES", "Y"]:
    print("nice")