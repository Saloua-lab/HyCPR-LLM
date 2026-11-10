# Conformal prediction: ordinal sets, conditional coverage, Mondrian CP (split CP, 1,000 splits)

## mistral LDOS-H eps=0.1: coverage 90.3%, width 2.60, set size 2.94 ratings
- user activity: high 90.8% (n=207), low 89.9% (n=245)
- item popularity: high 89.4% (n=193), low 91.0% (n=259)
- context (daytype): Holiday 86.6% (n=45), Weekend 90.1% (n=229), Working day 91.5% (n=178)
- true rating: 1-2 46.0% (n=42), 3 96.0% (n=108), 4-5 94.5% (n=302)
- predicted rating: 3-4 89.5% (n=233), <3 70.0% (n=30), >=4 94.5% (n=189)
- Mondrian on user activity: coverage 90.5%, width 2.63, set size 2.98; per group: high 90.3%, low 90.6%; by true rating: 1-2 48.1%, 3 95.9%, 4-5 94.5%
- Mondrian on item popularity: coverage 90.5%, width 2.64, set size 3.00; per group: high 90.5%, low 90.5%; by true rating: 1-2 47.3%, 3 96.1%, 4-5 94.6%
- Mondrian on predicted rating: coverage 91.0%, width 2.59, set size 3.00; per group: 3-4 90.6%, <3 95.0%, >=4 90.7%; by true rating: 1-2 55.2%, 3 89.5%, 4-5 96.6%

## mistral LDOS-H eps=0.05: coverage 95.5%, width 3.03, set size 3.49 ratings
- user activity: high 95.8% (n=207), low 95.2% (n=245)
- item popularity: high 93.9% (n=193), low 96.7% (n=259)
- context (daytype): Holiday 92.9% (n=45), Weekend 95.5% (n=229), Working day 96.1% (n=178)
- true rating: 1-2 70.1% (n=42), 3 99.7% (n=108), 4-5 97.5% (n=302)
- predicted rating: 3-4 96.3% (n=233), <3 80.3% (n=30), >=4 96.9% (n=189)
- Mondrian on user activity: coverage 95.6%, width 3.07, set size 3.54; per group: high 95.3%, low 95.8%; by true rating: 1-2 71.3%, 3 99.4%, 4-5 97.6%
- Mondrian on item popularity: coverage 95.7%, width 3.09, set size 3.56; per group: high 95.9%, low 95.6%; by true rating: 1-2 70.2%, 3 99.8%, 4-5 97.9%
- Mondrian on predicted rating: coverage 96.1%, width 3.01, set size 3.53; per group: 3-4 95.8%, <3 100.0%, >=4 95.8%; by true rating: 1-2 71.0%, 3 97.6%, 4-5 99.1%

## eurollm LDOS-H eps=0.1: coverage 90.2%, width 2.74, set size 3.09 ratings
- user activity: high 89.2% (n=207), low 91.1% (n=245)
- item popularity: high 88.0% (n=193), low 91.9% (n=259)
- context (daytype): Holiday 89.3% (n=45), Weekend 90.2% (n=229), Working day 90.5% (n=178)
- true rating: 1-2 54.9% (n=42), 3 97.1% (n=108), 4-5 92.7% (n=302)
- predicted rating: 3-4 90.8% (n=227), <3 77.9% (n=46), >=4 92.6% (n=179)
- Mondrian on user activity: coverage 90.4%, width 2.78, set size 3.14; per group: high 90.3%, low 90.5%; by true rating: 1-2 54.7%, 3 97.1%, 4-5 93.1%
- Mondrian on item popularity: coverage 90.5%, width 2.80, set size 3.16; per group: high 90.6%, low 90.4%; by true rating: 1-2 55.7%, 3 97.0%, 4-5 93.0%
- Mondrian on predicted rating: coverage 90.8%, width 2.77, set size 3.19; per group: 3-4 90.5%, <3 92.6%, >=4 90.8%; by true rating: 1-2 54.7%, 3 93.9%, 4-5 94.8%

## eurollm LDOS-H eps=0.05: coverage 95.5%, width 3.21, set size 3.68 ratings
- user activity: high 94.9% (n=207), low 95.9% (n=245)
- item popularity: high 93.7% (n=193), low 96.8% (n=259)
- context (daytype): Holiday 96.5% (n=45), Weekend 95.5% (n=229), Working day 95.1% (n=178)
- true rating: 1-2 74.2% (n=42), 3 99.9% (n=108), 4-5 96.8% (n=302)
- predicted rating: 3-4 97.8% (n=227), <3 82.1% (n=46), >=4 96.0% (n=179)
- Mondrian on user activity: coverage 95.5%, width 3.23, set size 3.71; per group: high 95.3%, low 95.7%; by true rating: 1-2 73.8%, 3 99.8%, 4-5 97.1%
- Mondrian on item popularity: coverage 95.7%, width 3.24, set size 3.73; per group: high 95.8%, low 95.6%; by true rating: 1-2 74.8%, 3 99.8%, 4-5 97.2%
- Mondrian on predicted rating: coverage 96.2%, width 3.21, set size 3.71; per group: 3-4 95.5%, <3 99.8%, >=4 96.1%; by true rating: 1-2 73.0%, 3 99.4%, 4-5 98.3%

## llama LDOS-H eps=0.1: coverage 90.3%, width 2.77, set size 3.09 ratings
- user activity: high 89.0% (n=207), low 91.4% (n=245)
- item popularity: high 86.7% (n=193), low 93.0% (n=259)
- context (daytype): Holiday 91.3% (n=45), Weekend 89.9% (n=229), Working day 90.5% (n=178)
- true rating: 1-2 55.5% (n=42), 3 97.1% (n=108), 4-5 92.7% (n=302)
- predicted rating: 3-4 90.5% (n=277), <3 75.0% (n=30), >=4 93.0% (n=145)
- Mondrian on user activity: coverage 90.4%, width 2.80, set size 3.13; per group: high 90.4%, low 90.5%; by true rating: 1-2 55.8%, 3 97.5%, 4-5 92.8%
- Mondrian on item popularity: coverage 90.5%, width 2.78, set size 3.11; per group: high 90.6%, low 90.4%; by true rating: 1-2 58.8%, 3 96.7%, 4-5 92.7%
- Mondrian on predicted rating: coverage 90.9%, width 2.78, set size 3.14; per group: 3-4 90.4%, <3 95.8%, >=4 90.8%; by true rating: 1-2 55.8%, 3 94.2%, 4-5 94.7%

## llama LDOS-H eps=0.05: coverage 95.5%, width 3.19, set size 3.64 ratings
- user activity: high 94.8% (n=207), low 96.2% (n=245)
- item popularity: high 93.6% (n=193), low 97.0% (n=259)
- context (daytype): Holiday 96.5% (n=45), Weekend 95.5% (n=229), Working day 95.4% (n=178)
- true rating: 1-2 72.4% (n=42), 3 99.9% (n=108), 4-5 97.2% (n=302)
- predicted rating: 3-4 97.2% (n=277), <3 79.2% (n=30), >=4 95.9% (n=145)
- Mondrian on user activity: coverage 95.6%, width 3.22, set size 3.68; per group: high 95.4%, low 95.8%; by true rating: 1-2 72.7%, 3 99.8%, 4-5 97.3%
- Mondrian on item popularity: coverage 95.7%, width 3.20, set size 3.66; per group: high 95.9%, low 95.6%; by true rating: 1-2 74.0%, 3 99.6%, 4-5 97.4%
- Mondrian on predicted rating: coverage 95.8%, width 3.17, set size 3.65; per group: 3-4 95.4%, <3 100.0%, >=4 95.9%; by true rating: 1-2 70.6%, 3 99.1%, 4-5 98.3%

## deepseek LDOS-H eps=0.1: coverage 90.3%, width 2.71, set size 3.03 ratings
- user activity: high 90.9% (n=207), low 89.8% (n=245)
- item popularity: high 88.8% (n=193), low 91.4% (n=259)
- context (daytype): Holiday 89.5% (n=45), Weekend 91.3% (n=229), Working day 89.1% (n=178)
- true rating: 1-2 49.1% (n=42), 3 98.0% (n=108), 4-5 93.3% (n=302)
- predicted rating: 3-4 88.3% (n=274), <3 82.2% (n=31), >=4 95.8% (n=147)
- Mondrian on user activity: coverage 90.5%, width 2.74, set size 3.06; per group: high 90.3%, low 90.6%; by true rating: 1-2 52.0%, 3 97.9%, 4-5 93.2%
- Mondrian on item popularity: coverage 90.5%, width 2.74, set size 3.07; per group: high 90.6%, low 90.5%; by true rating: 1-2 50.5%, 3 97.9%, 4-5 93.5%
- Mondrian on predicted rating: coverage 91.0%, width 2.72, set size 3.08; per group: 3-4 90.4%, <3 94.3%, >=4 91.2%; by true rating: 1-2 54.8%, 3 92.0%, 4-5 95.7%

## deepseek LDOS-H eps=0.05: coverage 95.5%, width 3.11, set size 3.55 ratings
- user activity: high 96.1% (n=207), low 95.0% (n=245)
- item popularity: high 94.7% (n=193), low 96.1% (n=259)
- context (daytype): Holiday 96.7% (n=45), Weekend 96.2% (n=229), Working day 94.4% (n=178)
- true rating: 1-2 72.9% (n=42), 3 99.6% (n=108), 4-5 97.2% (n=302)
- predicted rating: 3-4 95.9% (n=274), <3 85.0% (n=31), >=4 97.0% (n=147)
- Mondrian on user activity: coverage 95.6%, width 3.19, set size 3.64; per group: high 95.4%, low 95.7%; by true rating: 1-2 73.3%, 3 99.3%, 4-5 97.4%
- Mondrian on item popularity: coverage 95.7%, width 3.20, set size 3.66; per group: high 95.9%, low 95.6%; by true rating: 1-2 73.7%, 3 99.5%, 4-5 97.4%
- Mondrian on predicted rating: coverage 95.8%, width 3.13, set size 3.67; per group: 3-4 95.3%, <3 100.0%, >=4 95.9%; by true rating: 1-2 72.5%, 3 97.7%, 4-5 98.5%

## mistral Depaul-H eps=0.1: coverage 90.3%, width 2.70, set size 3.06 ratings
- user activity: high 90.6% (n=335), low 90.1% (n=349)
- item popularity: high 91.6% (n=341), low 89.0% (n=343)
- context (Location): Cinema 90.0% (n=354), Home 90.7% (n=330)
- true rating: 1-2 84.5% (n=220), 3 97.1% (n=140), 4-5 91.3% (n=324)
- predicted rating: 3-4 87.0% (n=190), <3 90.6% (n=270), >=4 92.7% (n=224)
- Mondrian on user activity: coverage 90.4%, width 2.69, set size 3.07; per group: high 90.4%, low 90.5%; by true rating: 1-2 84.9%, 3 97.2%, 4-5 91.3%
- Mondrian on item popularity: coverage 90.5%, width 2.67, set size 3.07; per group: high 90.4%, low 90.5%; by true rating: 1-2 85.2%, 3 96.4%, 4-5 91.5%
- Mondrian on predicted rating: coverage 90.7%, width 2.66, set size 3.10; per group: 3-4 91.0%, <3 90.5%, >=4 90.7%; by true rating: 1-2 86.0%, 3 93.6%, 4-5 92.7%

## mistral Depaul-H eps=0.05: coverage 95.2%, width 3.06, set size 3.62 ratings
- user activity: high 94.6% (n=335), low 95.8% (n=349)
- item popularity: high 95.6% (n=341), low 94.8% (n=343)
- context (Location): Cinema 95.5% (n=354), Home 94.9% (n=330)
- true rating: 1-2 91.1% (n=220), 3 100.0% (n=140), 4-5 96.0% (n=324)
- predicted rating: 3-4 95.1% (n=190), <3 95.2% (n=270), >=4 95.4% (n=224)
- Mondrian on user activity: coverage 95.5%, width 3.11, set size 3.67; per group: high 95.4%, low 95.5%; by true rating: 1-2 92.0%, 3 99.9%, 4-5 95.9%
- Mondrian on item popularity: coverage 95.5%, width 3.12, set size 3.67; per group: high 95.5%, low 95.5%; by true rating: 1-2 91.7%, 3 99.8%, 4-5 96.2%
- Mondrian on predicted rating: coverage 95.8%, width 3.16, set size 3.73; per group: 3-4 96.0%, <3 95.6%, >=4 95.7%; by true rating: 1-2 92.5%, 3 99.4%, 4-5 96.4%

## eurollm Depaul-H eps=0.1: coverage 90.4%, width 2.56, set size 2.91 ratings
- user activity: high 91.0% (n=335), low 89.8% (n=349)
- item popularity: high 91.8% (n=341), low 88.9% (n=343)
- context (Location): Cinema 89.7% (n=354), Home 91.1% (n=330)
- true rating: 1-2 87.8% (n=220), 3 96.0% (n=140), 4-5 89.7% (n=324)
- predicted rating: 3-4 87.4% (n=188), <3 88.9% (n=275), >=4 94.8% (n=221)
- Mondrian on user activity: coverage 90.5%, width 2.58, set size 2.94; per group: high 90.5%, low 90.5%; by true rating: 1-2 88.0%, 3 96.0%, 4-5 89.8%
- Mondrian on item popularity: coverage 90.5%, width 2.58, set size 2.95; per group: high 90.4%, low 90.5%; by true rating: 1-2 88.3%, 3 95.4%, 4-5 89.8%
- Mondrian on predicted rating: coverage 90.7%, width 2.56, set size 2.95; per group: 3-4 90.9%, <3 90.6%, >=4 90.7%; by true rating: 1-2 89.7%, 3 90.5%, 4-5 91.5%

## eurollm Depaul-H eps=0.05: coverage 95.2%, width 2.99, set size 3.47 ratings
- user activity: high 95.0% (n=335), low 95.5% (n=349)
- item popularity: high 96.5% (n=341), low 93.9% (n=343)
- context (Location): Cinema 95.8% (n=354), Home 94.7% (n=330)
- true rating: 1-2 93.3% (n=220), 3 99.7% (n=140), 4-5 94.6% (n=324)
- predicted rating: 3-4 95.9% (n=188), <3 93.6% (n=275), >=4 96.7% (n=221)
- Mondrian on user activity: coverage 95.4%, width 3.04, set size 3.53; per group: high 95.4%, low 95.5%; by true rating: 1-2 93.7%, 3 99.6%, 4-5 94.9%
- Mondrian on item popularity: coverage 95.5%, width 3.03, set size 3.52; per group: high 95.4%, low 95.6%; by true rating: 1-2 93.5%, 3 99.1%, 4-5 95.3%
- Mondrian on predicted rating: coverage 95.8%, width 3.06, set size 3.58; per group: 3-4 96.1%, <3 95.6%, >=4 95.6%; by true rating: 1-2 93.9%, 3 98.1%, 4-5 96.1%

## llama Depaul-H eps=0.1: coverage 90.4%, width 2.56, set size 2.91 ratings
- user activity: high 91.0% (n=335), low 89.8% (n=349)
- item popularity: high 92.5% (n=341), low 88.4% (n=343)
- context (Location): Cinema 89.6% (n=354), Home 91.3% (n=330)
- true rating: 1-2 87.0% (n=220), 3 97.0% (n=140), 4-5 89.9% (n=324)
- predicted rating: 3-4 85.7% (n=197), <3 89.5% (n=270), >=4 95.8% (n=217)
- Mondrian on user activity: coverage 90.5%, width 2.55, set size 2.91; per group: high 90.4%, low 90.5%; by true rating: 1-2 87.4%, 3 96.6%, 4-5 90.0%
- Mondrian on item popularity: coverage 90.5%, width 2.56, set size 2.93; per group: high 90.4%, low 90.7%; by true rating: 1-2 88.1%, 3 95.8%, 4-5 89.8%
- Mondrian on predicted rating: coverage 90.8%, width 2.53, set size 2.92; per group: 3-4 91.0%, <3 90.7%, >=4 90.7%; by true rating: 1-2 89.8%, 3 89.7%, 4-5 92.0%

## llama Depaul-H eps=0.05: coverage 95.2%, width 2.92, set size 3.39 ratings
- user activity: high 95.1% (n=335), low 95.4% (n=349)
- item popularity: high 96.8% (n=341), low 93.7% (n=343)
- context (Location): Cinema 95.3% (n=354), Home 95.2% (n=330)
- true rating: 1-2 93.8% (n=220), 3 99.6% (n=140), 4-5 94.4% (n=324)
- predicted rating: 3-4 95.3% (n=197), <3 93.8% (n=270), >=4 97.0% (n=217)
- Mondrian on user activity: coverage 95.5%, width 2.99, set size 3.47; per group: high 95.5%, low 95.5%; by true rating: 1-2 94.1%, 3 99.5%, 4-5 94.7%
- Mondrian on item popularity: coverage 95.5%, width 2.97, set size 3.44; per group: high 95.5%, low 95.5%; by true rating: 1-2 93.4%, 3 99.0%, 4-5 95.4%
- Mondrian on predicted rating: coverage 95.7%, width 3.00, set size 3.50; per group: 3-4 95.9%, <3 95.7%, >=4 95.7%; by true rating: 1-2 94.2%, 3 97.3%, 4-5 96.1%

## deepseek Depaul-H eps=0.1: coverage 90.4%, width 2.62, set size 2.95 ratings
- user activity: high 91.4% (n=335), low 89.4% (n=349)
- item popularity: high 91.7% (n=341), low 89.0% (n=343)
- context (Location): Cinema 90.0% (n=354), Home 90.8% (n=330)
- true rating: 1-2 86.9% (n=220), 3 95.7% (n=140), 4-5 90.4% (n=324)
- predicted rating: 3-4 88.0% (n=214), <3 88.5% (n=269), >=4 95.4% (n=201)
- Mondrian on user activity: coverage 90.5%, width 2.62, set size 2.96; per group: high 90.5%, low 90.5%; by true rating: 1-2 87.7%, 3 95.1%, 4-5 90.3%
- Mondrian on item popularity: coverage 90.5%, width 2.62, set size 2.98; per group: high 90.5%, low 90.6%; by true rating: 1-2 87.9%, 3 94.8%, 4-5 90.5%
- Mondrian on predicted rating: coverage 90.7%, width 2.58, set size 2.98; per group: 3-4 90.7%, <3 90.8%, >=4 90.6%; by true rating: 1-2 89.5%, 3 90.9%, 4-5 91.5%

## deepseek Depaul-H eps=0.05: coverage 95.2%, width 2.98, set size 3.47 ratings
- user activity: high 95.1% (n=335), low 95.3% (n=349)
- item popularity: high 96.3% (n=341), low 94.1% (n=343)
- context (Location): Cinema 95.0% (n=354), Home 95.3% (n=330)
- true rating: 1-2 93.6% (n=220), 3 99.8% (n=140), 4-5 94.3% (n=324)
- predicted rating: 3-4 95.4% (n=214), <3 93.5% (n=269), >=4 97.3% (n=201)
- Mondrian on user activity: coverage 95.4%, width 3.05, set size 3.54; per group: high 95.4%, low 95.5%; by true rating: 1-2 94.1%, 3 99.6%, 4-5 94.5%
- Mondrian on item popularity: coverage 95.5%, width 3.04, set size 3.53; per group: high 95.4%, low 95.6%; by true rating: 1-2 93.4%, 3 99.3%, 4-5 95.2%
- Mondrian on predicted rating: coverage 95.7%, width 3.06, set size 3.59; per group: 3-4 95.6%, <3 95.7%, >=4 95.6%; by true rating: 1-2 93.8%, 3 97.6%, 4-5 96.1%

## mistral Restaurant-H eps=0.1: coverage 90.8%, width 2.17, set size 2.54 ratings
- user activity: high 92.6% (n=125), low 89.1% (n=125)
- item popularity: high 87.5% (n=118), low 93.7% (n=132)
- context (Context): Weekday at Home 96.1% (n=45), Weekday at School 80.2% (n=48), Weekday at Work 93.9% (n=41), Weekend at Home 98.5% (n=33), Weekend at School 93.6% (n=55), Weekend at Work 81.6% (n=28)
- true rating: 1-2 82.9% (n=49), 3 88.0% (n=24), 4-5 93.4% (n=177)
- predicted rating: 3-4 84.1% (n=53), <3 82.8% (n=56), >=4 96.5% (n=141)
- Mondrian on user activity: coverage 91.3%, width 2.26, set size 2.64; per group: high 91.5%, low 91.1%; by true rating: 1-2 82.1%, 3 91.1%, 4-5 94.0%
- Mondrian on item popularity: coverage 91.3%, width 2.17, set size 2.59; per group: high 91.6%, low 91.0%; by true rating: 1-2 82.7%, 3 93.2%, 4-5 93.4%
- Mondrian on predicted rating: coverage 91.8%, width 2.05, set size 2.50; per group: 3-4 93.1%, <3 93.2%, >=4 90.8%; by true rating: 1-2 90.6%, 3 79.4%, 4-5 93.9%

## mistral Restaurant-H eps=0.05: coverage 96.1%, width 2.81, set size 3.32 ratings
- user activity: high 97.3% (n=125), low 94.9% (n=125)
- item popularity: high 95.2% (n=118), low 96.9% (n=132)
- context (Context): Weekday at Home 99.7% (n=45), Weekday at School 88.6% (n=48), Weekday at Work 98.3% (n=41), Weekend at Home 100.0% (n=33), Weekend at School 97.2% (n=55), Weekend at Work 93.0% (n=28)
- true rating: 1-2 90.9% (n=49), 3 98.5% (n=24), 4-5 97.2% (n=177)
- predicted rating: 3-4 94.9% (n=53), <3 91.6% (n=56), >=4 98.3% (n=141)
- Mondrian on user activity: coverage 96.6%, width 3.03, set size 3.56; per group: high 96.6%, low 96.5%; by true rating: 1-2 92.3%, 3 98.6%, 4-5 97.5%
- Mondrian on item popularity: coverage 96.3%, width 2.94, set size 3.47; per group: high 96.7%, low 96.0%; by true rating: 1-2 90.8%, 3 97.8%, 4-5 97.7%
- Mondrian on predicted rating: coverage 97.1%, width 2.75, set size 3.39; per group: 3-4 99.0%, <3 98.9%, >=4 95.6%; by true rating: 1-2 95.3%, 3 87.5%, 4-5 99.0%

## eurollm Restaurant-H eps=0.1: coverage 90.8%, width 2.18, set size 2.55 ratings
- user activity: high 91.7% (n=125), low 89.9% (n=125)
- item popularity: high 86.3% (n=118), low 94.9% (n=132)
- context (Context): Weekday at Home 95.2% (n=45), Weekday at School 81.3% (n=48), Weekday at Work 94.2% (n=41), Weekend at Home 94.0% (n=33), Weekend at School 94.3% (n=55), Weekend at Work 84.5% (n=28)
- true rating: 1-2 83.3% (n=49), 3 85.4% (n=24), 4-5 93.6% (n=177)
- predicted rating: 3-4 83.5% (n=59), <3 83.4% (n=52), >=4 96.7% (n=139)
- Mondrian on user activity: coverage 91.3%, width 2.25, set size 2.65; per group: high 91.5%, low 91.2%; by true rating: 1-2 83.6%, 3 87.6%, 4-5 94.0%
- Mondrian on item popularity: coverage 91.2%, width 2.12, set size 2.52; per group: high 91.6%, low 90.9%; by true rating: 1-2 83.8%, 3 91.7%, 4-5 93.2%
- Mondrian on predicted rating: coverage 91.7%, width 2.05, set size 2.49; per group: 3-4 93.0%, <3 93.2%, >=4 90.6%; by true rating: 1-2 91.4%, 3 79.1%, 4-5 93.6%

## eurollm Restaurant-H eps=0.05: coverage 96.1%, width 2.80, set size 3.29 ratings
- user activity: high 96.6% (n=125), low 95.6% (n=125)
- item popularity: high 94.7% (n=118), low 97.4% (n=132)
- context (Context): Weekday at Home 99.0% (n=45), Weekday at School 91.1% (n=48), Weekday at Work 97.5% (n=41), Weekend at Home 98.8% (n=33), Weekend at School 97.0% (n=55), Weekend at Work 93.2% (n=28)
- true rating: 1-2 91.3% (n=49), 3 98.1% (n=24), 4-5 97.2% (n=177)
- predicted rating: 3-4 93.6% (n=59), <3 91.4% (n=52), >=4 98.9% (n=139)
- Mondrian on user activity: coverage 96.5%, width 2.99, set size 3.50; per group: high 96.5%, low 96.4%; by true rating: 1-2 92.7%, 3 97.7%, 4-5 97.4%
- Mondrian on item popularity: coverage 96.3%, width 2.84, set size 3.34; per group: high 96.6%, low 96.0%; by true rating: 1-2 90.3%, 3 97.3%, 4-5 97.9%
- Mondrian on predicted rating: coverage 97.0%, width 2.70, set size 3.32; per group: 3-4 98.2%, <3 99.2%, >=4 95.6%; by true rating: 1-2 96.2%, 3 85.3%, 4-5 98.9%

## llama Restaurant-H eps=0.1: coverage 90.8%, width 2.17, set size 2.53 ratings
- user activity: high 91.8% (n=125), low 89.9% (n=125)
- item popularity: high 87.3% (n=118), low 94.0% (n=132)
- context (Context): Weekday at Home 96.7% (n=45), Weekday at School 81.4% (n=48), Weekday at Work 94.4% (n=41), Weekend at Home 93.5% (n=33), Weekend at School 93.5% (n=55), Weekend at Work 84.0% (n=28)
- true rating: 1-2 82.9% (n=49), 3 87.4% (n=24), 4-5 93.5% (n=177)
- predicted rating: 3-4 84.2% (n=56), <3 83.4% (n=54), >=4 96.4% (n=140)
- Mondrian on user activity: coverage 91.3%, width 2.25, set size 2.63; per group: high 91.5%, low 91.1%; by true rating: 1-2 82.6%, 3 89.5%, 4-5 94.0%
- Mondrian on item popularity: coverage 91.2%, width 2.15, set size 2.55; per group: high 91.5%, low 90.9%; by true rating: 1-2 83.1%, 3 92.9%, 4-5 93.3%
- Mondrian on predicted rating: coverage 91.8%, width 2.05, set size 2.45; per group: 3-4 93.0%, <3 93.0%, >=4 90.8%; by true rating: 1-2 90.2%, 3 79.5%, 4-5 93.9%

## llama Restaurant-H eps=0.05: coverage 96.1%, width 2.80, set size 3.30 ratings
- user activity: high 96.6% (n=125), low 95.6% (n=125)
- item popularity: high 94.8% (n=118), low 97.3% (n=132)
- context (Context): Weekday at Home 99.2% (n=45), Weekday at School 90.8% (n=48), Weekday at Work 97.2% (n=41), Weekend at Home 99.4% (n=33), Weekend at School 96.6% (n=55), Weekend at Work 93.6% (n=28)
- true rating: 1-2 90.5% (n=49), 3 98.4% (n=24), 4-5 97.3% (n=177)
- predicted rating: 3-4 95.0% (n=56), <3 91.6% (n=54), >=4 98.3% (n=140)
- Mondrian on user activity: coverage 96.5%, width 3.00, set size 3.52; per group: high 96.6%, low 96.4%; by true rating: 1-2 92.3%, 3 98.0%, 4-5 97.5%
- Mondrian on item popularity: coverage 96.3%, width 2.87, set size 3.38; per group: high 96.6%, low 96.0%; by true rating: 1-2 90.2%, 3 97.3%, 4-5 97.9%
- Mondrian on predicted rating: coverage 97.1%, width 2.76, set size 3.41; per group: 3-4 98.7%, <3 98.9%, >=4 95.7%; by true rating: 1-2 95.0%, 3 87.8%, 4-5 99.0%

## deepseek Restaurant-H eps=0.1: coverage 90.8%, width 2.18, set size 2.55 ratings
- user activity: high 91.5% (n=125), low 90.1% (n=125)
- item popularity: high 86.7% (n=118), low 94.4% (n=132)
- context (Context): Weekday at Home 96.1% (n=45), Weekday at School 80.7% (n=48), Weekday at Work 94.9% (n=41), Weekend at Home 93.7% (n=33), Weekend at School 94.3% (n=55), Weekend at Work 82.9% (n=28)
- true rating: 1-2 83.0% (n=49), 3 86.6% (n=24), 4-5 93.5% (n=177)
- predicted rating: 3-4 83.5% (n=57), <3 84.1% (n=53), >=4 96.3% (n=140)
- Mondrian on user activity: coverage 91.3%, width 2.27, set size 2.66; per group: high 91.5%, low 91.1%; by true rating: 1-2 83.3%, 3 89.0%, 4-5 93.9%
- Mondrian on item popularity: coverage 91.3%, width 2.18, set size 2.57; per group: high 91.6%, low 91.0%; by true rating: 1-2 83.8%, 3 93.2%, 4-5 93.1%
- Mondrian on predicted rating: coverage 91.8%, width 2.09, set size 2.50; per group: 3-4 92.6%, <3 93.5%, >=4 90.9%; by true rating: 1-2 89.6%, 3 79.5%, 4-5 94.1%

## deepseek Restaurant-H eps=0.05: coverage 96.1%, width 2.82, set size 3.32 ratings
- user activity: high 96.5% (n=125), low 95.8% (n=125)
- item popularity: high 94.8% (n=118), low 97.3% (n=132)
- context (Context): Weekday at Home 98.6% (n=45), Weekday at School 90.2% (n=48), Weekday at Work 97.7% (n=41), Weekend at Home 99.5% (n=33), Weekend at School 97.7% (n=55), Weekend at Work 92.8% (n=28)
- true rating: 1-2 90.7% (n=49), 3 98.4% (n=24), 4-5 97.3% (n=177)
- predicted rating: 3-4 94.8% (n=57), <3 91.8% (n=53), >=4 98.3% (n=140)
- Mondrian on user activity: coverage 96.5%, width 3.02, set size 3.54; per group: high 96.5%, low 96.4%; by true rating: 1-2 92.2%, 3 98.0%, 4-5 97.5%
- Mondrian on item popularity: coverage 96.3%, width 2.88, set size 3.39; per group: high 96.7%, low 95.9%; by true rating: 1-2 90.4%, 3 97.7%, 4-5 97.8%
- Mondrian on predicted rating: coverage 97.1%, width 2.79, set size 3.43; per group: 3-4 98.4%, <3 99.2%, >=4 95.7%; by true rating: 1-2 94.6%, 3 87.9%, 4-5 99.1%
