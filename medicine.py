age = int(input("Enter your age: "))
weight = int (input("weight(kg):"))
if age < 15 and weight <55:
    print("sorry this medicine is not given to you,")
elif weight>=55 and age >=18:
    print("this medicine is for you.")
else:
    print( "sorry, you can't take this medicine.")

