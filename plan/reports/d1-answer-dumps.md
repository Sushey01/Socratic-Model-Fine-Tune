# D1 — explain/tell-me → definition dump

Listed only. Train JSONL was **not** edited.

## Counts

- Train ASK-like student turns: **985**
- Train flagged as direct answer (no `?`, not a short nudge): **378**
- Train ASK-like still Socratic: **607**
- Unique opening user lines among train dumps: **105**
- Annotated ASK-like: **1091**; flagged dumps: **426**
- Full list: `plan/reports/d1-answer-dumps-train.jsonl`

Annotated dump strategies: CLARIFY=415, REDIRECT=11

## Example dumps (train)

### Train line 43
- Opening: 'How do we know which metal is more reactive than another?'
- Student: 'Can you break that down more?'
- Assistant: 'In a displacement reaction, a more reactive metal displaces a less reactive metal from its compound in solution. Since iron displaces copper from copper sulphate solution, this shows iron is more reactive than copper, consistent with its higher position in the reactivity series.'

### Train line 51
- Opening: 'Why does dropping a metal into acid produce bubbles?'
- Student: 'Can you explain that further?'
- Assistant: "When a reactive metal reacts with a dilute acid, it displaces hydrogen, producing hydrogen gas and a salt. For example, zinc reacting with hydrochloric acid produces zinc chloride and hydrogen gas, which can be identified by the characteristic 'pop' sound when a burning splint is brought near it."

### Train line 69
- Opening: 'How does a generator produce electricity?'
- Student: 'Can you explain that further?'
- Assistant: 'Electromagnetic induction occurs when a changing magnetic field through a coil induces an electric current in it. This only happens while the field is changing — for example, while a magnet is moving relative to the coil — which is the principle generators use to produce electricity.'

### Train line 90
- Opening: 'What is puberty?'
- Student: 'What does that mean exactly?'
- Assistant: 'Puberty is triggered by increased levels of hormones — testosterone in males and oestrogen in females — released by the reproductive organs, which cause the physical and reproductive changes associated with sexual maturity.'

### Train line 92
- Opening: "Why do some wires get hot and others don't?"
- Student: "I'm still confused, could you go deeper?"
- Assistant: 'Resistance is the opposition a material offers to the flow of electric current, caused by collisions between moving electrons and the atoms of the conductor, which converts some electrical energy into heat.'

### Train line 116
- Opening: 'What is resistance?'
- Student: 'Can you explain that further?'
- Assistant: 'Resistance is the opposition a material offers to the flow of electric current, caused by collisions between moving electrons and the atoms of the conductor, which converts some electrical energy into heat.'

### Train line 132
- Opening: 'Why do we breathe?'
- Student: 'What does that mean exactly?'
- Assistant: "Respiration is the process where cells break down glucose using oxygen to release energy, producing carbon dioxide and water as waste. That's why breathing rate increases during exercise — your muscles need more oxygen to release more energy."

### Train line 147
- Opening: 'How does the eye focus on both near and far objects?'
- Student: "I'm still confused, could you go deeper?"
- Assistant: "Accommodation is the eye's ability to adjust the curvature of its lens using the ciliary muscles, changing its focal length so that images of objects at varying distances are focused sharply on the retina."

## Example dumps (annotated, with concept)

### conv_00071 — Life Processes / Nutrition in plants (`CLARIFY`)
- Student: 'Can you explain that further?'
- Tutor: 'Plants make their own food through photosynthesis: chlorophyll in their leaves absorbs sunlight and uses that energy to combine carbon dioxide and water into glucose, releasing oxygen as a byproduct.'

### conv_00072 — Life Processes / Nutrition in plants (`CLARIFY`)
- Student: "I'm still confused, could you go deeper?"
- Tutor: 'Plants make their own food through photosynthesis: chlorophyll in their leaves absorbs sunlight and uses that energy to combine carbon dioxide and water into glucose, releasing oxygen as a byproduct.'

### conv_00073 — Life Processes / Nutrition in plants (`CLARIFY`)
- Student: 'Could you elaborate on that?'
- Tutor: 'Plants make their own food through photosynthesis: chlorophyll in their leaves absorbs sunlight and uses that energy to combine carbon dioxide and water into glucose, releasing oxygen as a byproduct.'

### conv_00074 — Life Processes / Nutrition in plants (`CLARIFY`)
- Student: 'Could you elaborate on that?'
- Tutor: 'Plants make their own food through photosynthesis: chlorophyll in their leaves absorbs sunlight and uses that energy to combine carbon dioxide and water into glucose, releasing oxygen as a byproduct.'

### conv_00075 — Life Processes / Nutrition in plants (`CLARIFY`)
- Student: "I'm still confused, could you go deeper?"
- Tutor: 'Plants make their own food through photosynthesis: chlorophyll in their leaves absorbs sunlight and uses that energy to combine carbon dioxide and water into glucose, releasing oxygen as a byproduct.'

### conv_00077 — Life Processes / Nutrition in plants (`CLARIFY`)
- Student: 'Can you explain that further?'
- Tutor: 'Plants make their own food through photosynthesis: chlorophyll in their leaves absorbs sunlight and uses that energy to combine carbon dioxide and water into glucose, releasing oxygen as a byproduct.'

## Concepts with the most dumps (annotated)

- 18  Scattering of light and why the sky is blue
- 13  Puberty and reproductive maturity
- 11  Newton's first law (inertia)
- 10  Refraction and bending of light
- 10  Periodic trends in groups
- 10  Homologous series
- 9  Nutrition in plants
- 9  Reflex action
- 9  Electric current
- 9  Resistance
- 9  Sex determination in humans
- 9  Pressure and surface area
- 9  Longitudinal vs transverse waves
- 9  Biodegradable vs non-biodegradable waste
- 9  Water of crystallization
- 8  Respiration
- 8  Excretion
- 8  Hormones and the endocrine system
- 8  Oxidation and corrosion
- 8  Electromagnetic induction

Wait before removing or rewriting any of these.