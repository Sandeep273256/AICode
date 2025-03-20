def grade_eligibility(score):
    if score >= 90:
        return "great you score Grade A."
    elif score >= 80:
        return " good you score Grade B."
    elif score >= 70:
        return "you score Grade C."
    elif score >= 60:
        return " you score Grade D."
    else:
        return " you score Grade E. "
score = int(input ("please enter you score:"))
result = (grade_eligibility(score))
print(result)