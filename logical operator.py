age = int(input("Enter your age: "))
nationality = input("Are you a citizen?  (yes/no): ").lower()
if age < 18:
    print("sorry you are under age")
elif nationality=="yes" and age >=18:
    print("congratulations you are eligible for voting.")
else:
    print( "sorry, you have not fulfill the requirements.")

