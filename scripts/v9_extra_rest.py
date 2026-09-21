"""More v9 extras. Last 3 of each list = holdout."""

from gold_v8_more import _p

def _h(rows):
    """Prefix last 3 questions with HOLDOUT: if missing."""
    out = []
    n = len(rows)
    for i, r in enumerate(rows):
        d = dict(r)
        if i >= n - 3 and not d["q"].startswith("HOLDOUT:"):
            d["q"] = "HOLDOUT: " + d["q"]
        out.append(d)
    return out


V9_EXTRA_REST: dict[str, list[dict[str, str]]] = {}


def add(name: str, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
    V9_EXTRA_REST[name] = _h(_p(rows))


add("Ohm's Law", [
    ("What is Ohm's law?", "If the resistor stays the same, does current go up when voltage goes up?", "Same pipe, stronger push. More flow?", "Is that V, I, and R hanging together?", "Ohm's law is about plant food.", "Current goes up with voltage if R is fixed.", "That's it — that's Ohm's law."),
    ("What is voltage, in a circuit picture?", "Is it the push on charge, or the flow itself?", "Battery as a pump. Push vs flow?", "Not the same as current?", "Voltage is how hot the wire is.", "The push on charge.", "That's it — voltage is the push."),
    ("What is current here?", "Charge passing per second, or the push?", "Amperes. Flow rate?", "Not volts?", "Current is the colour of copper.", "Charge per second.", "That's it — current is flow of charge."),
    ("What is resistance?", "Opposition to flow, often making heat?", "A narrow pipe. Harder flow?", "R in V=IR?", "Resistance is a type of voltage.", "Opposition to current.", "That's it — resistance opposes flow."),
    ("What does V = IR say in words?", "Voltage equals current times resistance?", "If I double and R fixed, V doubles?", "A product, not a sum?", "V equals I divided by R always.", "V is I times R.", "That's it — you stated V=IR."),
    ("What stays fixed in a simple Ohm's-law demo?", "The resistor, while we change V or I?", "One component's R. Fixed?", "Not the whole universe?", "Everything changes including R always.", "The resistance of that component.", "That's it — R of that part is the fixed idea."),
    ("Why not treat V, I, R as unrelated letters?", "Do they describe one situation together?", "Change one, another shifts. Linked?", "That's the law's point?", "They are random exam letters.", "They belong to one circuit situation.", "That's it — they are linked."),
    ("What is an ohm?", "A unit of resistance?", "Not a unit of time?", "R measured in ohms?", "An ohm is a type of current.", "The unit of resistance.", "That's it — ohm is resistance's unit."),
    ("Does Ohm's law care about a broken circuit?", "If I is zero because the loop is open, is that a different issue than R?", "Open switch. No flow?", "Law assumes a complete path?", "Open circuits still have huge current.", "No path means no current; that's the break.", "That's it — you need a closed loop first."),
    ("What happens to I if R doubles and V stays the same?", "Does current fall to about half?", "Harder path, same push. Less flow?", "I = V/R with R×2?", "Current doubles if R doubles.", "Current about halves.", "That's it — bigger R, smaller I if V is fixed."),
])

add("Nutrition in plants", [
    ("What is nutrition in a plant?", "Is it how the plant gets materials and energy to live, without chewing meals?", "Making food plus taking water/minerals. Nutrition?", "Not the same as an animal eating meat?", "Plant nutrition means watching TV.", "How it obtains food and materials to live.", "That's it — that's plant nutrition."),
    ("What is photosynthesis, in one student line?", "Do leaves use light to build food from CO2 and water?", "Green leaves, sunlight. Build glucose?", "Stay on plant food-making.", "Photosynthesis is animal digestion.", "Using light to make food from CO2 and water.", "That's it — that's photosynthesis."),
    ("What is chlorophyll?", "The green pigment that traps light for that food-making?", "Why leaves look green. Pigment?", "Not a type of root?", "Chlorophyll is a soil vitamin.", "The green light-trapping pigment.", "That's it — chlorophyll traps light."),
    ("What is an autotroph?", "An organism that makes its own food?", "Plant vs you. Self-feeder?", "Opposite of eating others?", "Autotrophs hunt deer.", "A self-feeder / food-maker.", "That's it — autotroph makes its own food."),
    ("What gas does a plant take in for photosynthesis?", "Carbon dioxide, or mostly nitrogen?", "From air into the leaf. Which carbon gas?", "Not oxygen as the main ingredient here?", "They inhale only nitrogen for food.", "Carbon dioxide.", "That's it — CO2 is the carbon source."),
    ("What does the plant give out that we often mention?", "Oxygen as a byproduct, in the school picture?", "Bubbles on pondweed. O2?", "Still about plant food-making, not sound.", "They give out nitrogen only.", "Oxygen.", "That's it — oxygen is released."),
    ("What is a heterotroph?", "Must eat others' food, unlike a green plant?", "You vs grass. Who makes food?", "Not a type of mineral?", "Heterotrophs are only rocks.", "An eater of others' food.", "That's it — heterotrophs don't make that food."),
    ("What is starch doing in a leaf test?", "Stored food made after photosynthesis?", "Iodine test. Starch as evidence?", "Not a type of plastic?", "Starch is a waste gas.", "Stored carbohydrate from photosynthesis.", "That's it — starch shows food was made."),
    ("Why water the soil if food is made in leaves?", "Do roots still take water as a raw material and for transport?", "Photosynthesis needs water. Roots?", "Water isn't the 'meal' but an ingredient?", "Water is unused decoration.", "Water is an ingredient and for transport.", "That's it — water still matters."),
    ("What is a producer in a plant sense?", "The plant making food for the chain?", "Same as autotroph here?", "Not a factory worker?", "Producers are animals at the top.", "The plant that makes food.", "That's it — the plant is the producer."),
])

add("Respiration", [
    ("What is respiration in Grade 10?", "Is it cells releasing energy from food, often using oxygen?", "Not only 'breathing in and out'. Cells?", "Stay on energy from food.", "Respiration is a type of light.", "Releasing energy from food in cells.", "That's it — respiration is energy from food in cells."),
    ("What is breathing then?", "Moving air in and out so gas exchange can happen?", "Lungs vs mitochondria story. Different words?", "Breathing helps respiration, not identical?", "Breathing is photosynthesis.", "Air in and out; not the whole energy release.", "That's it — you unmixed breathing and respiration."),
    ("What gas do we take in for aerobic respiration?", "Oxygen?", "Not nitrogen as the fuel gas?", "Stay on this process.", "We respire helium.", "Oxygen.", "That's it — oxygen for aerobic respiration."),
    ("What gas do we breathe out extra of?", "Carbon dioxide from breaking down food?", "Limewater test. CO2?", "Waste of respiration?", "We breathe out only oxygen.", "Carbon dioxide.", "That's it — extra CO2 out."),
    ("What is aerobic meaning here?", "Using oxygen, versus without?", "With air-ish. Oxygen path?", "Opposite of yeast fermentation in bread?", "Aerobic means happening on the moon only.", "With oxygen.", "That's it — aerobic uses oxygen."),
    ("What is anaerobic respiration, school-level?", "Energy from food without oxygen, like in yeast or tired muscle?", "Less energy, different products. Without O2?", "Not a type of light wave?", "Anaerobic means extra oxygen.", "Without oxygen.", "That's it — anaerobic is without oxygen."),
    ("What is glucose doing in respiration?", "The food molecule being broken for energy?", "Fuel. Glucose?", "Stay on energy release.", "Glucose is a type of oxygen.", "The fuel being broken down.", "That's it — glucose is the fuel."),
    ("What is oxygen's job in aerobic respiration?", "Helps break food and release energy, making CO2 and water in the school equation?", "Not to paint blood blue?", "Reactant, not a vitamin?", "Oxygen stores fat.", "It helps release energy from food.", "That's it — oxygen helps release the energy."),
    ("Why does exercise raise breathing rate?", "Muscles need more energy, so more gas exchange?", "More respiration. More air?", "Stay on this, not on light.", "Exercise fills lungs with food.", "More energy need, more breathing.", "That's it — more demand, more breathing."),
    ("What is a mitochondrion, in one line?", "The cell part often called the site of respiration?", "Energy release location. Organelle?", "Not a type of bone?", "Mitochondria are only in plants' leaves as green paint.", "Where a lot of respiration happens in the cell.", "That's it — that's the school role of mitochondria."),
])
