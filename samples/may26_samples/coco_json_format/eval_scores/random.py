unguided = {"Bleu_1": 0.613261878099455,
  "Bleu_2": 0.4236994116195719,
  "Bleu_3": 0.2800586204511092,
  "Bleu_4": 0.1834560220820554,}

guided = {
    "Bleu_1": 0.5080651569328162,
  "Bleu_2": 0.31911174929281383,
  "Bleu_3": 0.1918507916613567,
  "Bleu_4": 0.11661196075353769,
}

gaussian = {
    "Bleu_1": 0.6050253646163122,
  "Bleu_2": 0.4203261512374819,
  "Bleu_3": 0.2813342942610044,
  "Bleu_4": 0.18592201527493646,
}
sum=0
for x in unguided :
    sum += unguided[x]

print(sum / 4)
sum=0
for x in guided :
    sum += guided[x]

print(sum / 4)
sum=0
for x in gaussian :
    sum += gaussian[x]

print(sum / 4)