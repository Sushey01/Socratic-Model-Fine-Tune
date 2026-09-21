"""v9 extras for remaining concepts. Last 3 holdout via _h."""

from v9_extra_rest import add, V9_EXTRA_REST

# add() writes into V9_EXTRA_REST

add("Electric current", [
    ("What is electric current?", "Is it charge moving through the metal, or the plastic coating flowing?", "Mobile electrons in metal. What moves?", "Stay on the wire, not on light.", "Current is melted plastic.", "Charge moving in the conductor.", "That's it — current is moving charge."),
    ("What is a circuit?", "A closed loop so charge can go out and back?", "Broken ring. Can it flow?", "Need a complete path?", "A circuit is a type of plant.", "A closed path for current.", "That's it — a complete loop."),
    ("What is an ampere?", "A unit of current, charge per time?", "Not a unit of mass?", "A for current?", "An ampere measures temperature.", "The unit of current.", "That's it — ampere is current's unit."),
    ("What is conventional current?", "The arrow we draw, even if electrons go the other way?", "A label. Physics of heat still from motion?", "Convention vs electron direction?", "Conventional current is a type of acid.", "The usual arrow for current direction.", "That's it — it's the drawing convention."),
    ("What does a battery do in a simple circuit?", "Provide the push (voltage) that can drive current if the loop is closed?", "Energy source for the push. Battery?", "Not a magnet that eats charge?", "Batteries store light.", "It provides the push for charge.", "That's it — the battery is the push source."),
    ("What is a conductor?", "A material that lets charge flow easily?", "Copper vs rubber. Who conducts?", "Stay on current.", "Conductors are only gases.", "A material that lets current through easily.", "That's it — that's a conductor."),
    ("What is an insulator?", "A material that strongly resists current?", "Plastic coating. Why it's there?", "Opposite of conductor?", "Insulators are the best wires.", "It blocks current.", "That's it — insulator resists flow."),
    ("Why series current is the same through each bulb?", "One loop, one flow rate?", "Hose loop. Same I?", "Not used up like fuel?", "The last bulb gets zero always.", "Same current around one loop.", "That's it — series current is the same."),
    ("What is static charge versus current?", "Stuck charge vs charge on the move?", "Hair and a comb vs a working bulb. Different?", "Current needs a path and a push over time?", "They are identical.", "Current is charge moving; static is sitting charge.", "That's it — you unmixed static and current."),
    ("What happens if the switch is open?", "The loop is broken so current stops?", "Gap. Flow?", "Stay on this circuit.", "Open switch means more current.", "No complete path, no current.", "That's it — open means off."),
])

add("Resistance", [
    ("What is electrical resistance?", "Opposition to current, often with heating?", "Collisions in the metal. Opposition?", "Not a type of voltage?", "Resistance is a smell.", "Opposition to current.", "That's it — that's resistance."),
    ("What is a resistor?", "A component we put in on purpose to limit current?", "A designed R. Tool?", "Not only a nuisance?", "Resistors store water.", "A part that provides resistance.", "That's it — a resistor is that component."),
    ("What happens to R if a wire is longer?", "More obstacle course, R up?", "Length. Harder path?", "Same metal, longer — more R?", "Longer wire has zero R.", "Resistance increases.", "That's it — longer wire, more R."),
    ("What happens to R if the wire is thicker?", "Easier path, R down?", "Width. Corridor?", "Same length, thicker — less R?", "Thicker always means more R.", "Resistance decreases.", "That's it — thicker wire, less R."),
    ("Why do resistors get warm?", "Energy transferred to heat as charge is opposed?", "Collisions. Heat?", "Stay on resistance, not on sound.", "Heat means R is zero.", "Opposition dumps energy as heat.", "That's it — resistance can heat the part."),
    ("What is resistivity, school-level?", "A material property, how strongly that substance resists, not just the object's shape?", "Copper vs nichrome. Material?", "Different from R of one piece?", "Resistivity is a type of current.", "How the material itself resists.", "That's it — resistivity is the material's tendency."),
    ("Does temperature affect metal R?", "Usually R up when hotter for metals?", "Vibrating atoms. Harder path?", "Typical metals?", "Heat always removes all R.", "Metal resistance often rises with temperature.", "That's it — hotter metal, typically more R."),
    ("What is a short circuit, in R terms?", "A path with very low R so huge current can flow?", "Danger. Tiny R?", "Not a type of plant?", "Short circuit means no electricity exists.", "A very low-R path.", "That's it — short means too little resistance."),
    ("Why put R in series with an LED?", "To limit current so the LED survives?", "Designed limit. Purpose?", "Not to make light from the resistor mainly?", "LEDs need infinite current.", "To keep current in a safe range.", "That's it — R protects by limiting I."),
    ("Is R the same as voltage?", "No — R is opposition, V is push?", "Different jobs. Unmix?", "V=IR links them, they're not identical?", "R and V are the same unit.", "No — different quantities.", "That's it — you unmixed R and V."),
])

add("Refraction and bending of light", [
    ("What is refraction?", "Light changing direction as it goes from air into water or glass?", "Straw looking bent. Path change?", "Stay on light, not sound in air as the topic.", "Refraction is a type of magnet.", "Light bending at a transparent boundary.", "That's it — refraction is that bend."),
    ("What is a medium, here?", "The material light is travelling in, like air or water?", "From air to water. Two media?", "Not a news website?", "A medium is a type of acid.", "The material light is in.", "That's it — medium means the material."),
    ("What is the normal in a refraction diagram?", "A line perpendicular to the surface?", "Dotted line. Reference for angles?", "Not a battery?", "The normal is a type of current.", "The perpendicular to the surface.", "That's it — that's the normal."),
    ("When does light bend toward the normal?", "When it slows, like air into glass in the school picture?", "Faster to slower. Toward?", "Stay on light speed in media.", "It always bends away no matter what.", "When it goes into a slower medium.", "That's it — slower medium, toward the normal."),
    ("What is apparent depth?", "The pool looking shallower because rays bend as they leave water?", "Your eye assumes straight rays. Trick?", "Still light, not sound?", "The pool actually shrinks.", "A refraction trick that raises the bottom.", "That's it — apparent depth is that trick."),
    ("Does light along the normal bend aside?", "No sideways bend, maybe only speed change?", "Hit straight on. Deflect?", "Special case?", "It must zigzag always.", "No sideways bend along the normal.", "That's it — along the normal, no side bend."),
    ("What is the difference between refraction and reflection?", "One goes through and bends; one bounces?", "Window vs mirror. Different?", "Stay on light.", "They are the same word.", "Refraction transmits with a bend; reflection bounces.", "That's it — you split bounce vs bend-through."),
    ("Why a lens can focus light?", "Refraction at curved surfaces changing ray directions?", "Lens as shaped glass. Bend?", "Not because glass is sticky?", "Lenses work by magnetism.", "Curved surfaces refract rays to a focus.", "That's it — focusing is organised refraction."),
    ("What is optical density, loosely?", "How much a medium slows light, in school talk?", "Glass vs air. Slower in glass?", "Related to the bend?", "Optical density is mass of the table.", "How strongly the medium slows light.", "That's it — that's the school meaning."),
    ("Can refraction happen in a window you can see through?", "Yes — transparent boundary, not only a prism show?", "Everyday glass. Still refraction?", "Not only rainbows?", "Windows never refract.", "Yes, at everyday glass too.", "That's it — refraction is common at glass."),
])

add("Laws of reflection", [
    ("What is reflection of light?", "Light bouncing off a surface back into the same medium?", "Mirror. Bounce?", "Stay on light.", "Reflection is a plant process.", "Light bouncing off a surface.", "That's it — that's reflection."),
    ("What is the angle of incidence?", "The incoming angle measured from the normal?", "i in the diagram. From the perpendicular?", "Not from the mirror face in the law?", "Incidence is a type of current.", "Incoming angle from the normal.", "That's it — that's i."),
    ("What is the angle of reflection?", "The outgoing bounce, also from the normal?", "r. Equal to i?", "The law's pair?", "Reflection angle is temperature.", "The bounce angle from the normal.", "That's it — that's r."),
    ("What is the law of reflection?", "i equals r, and the rays sit in one plane with the normal?", "Equal angles. Plane?", "Not that light is eaten?", "The law says light never bounces.", "Angle in equals angle out (from the normal).", "That's it — that's the law."),
    ("What is a regular reflection?", "Orderly bounce from a smooth surface, a clear image?", "Mirror vs paper. Smooth?", "Sharp image?", "Regular means the mirror is late.", "Smooth surface, ordered bounce.", "That's it — regular reflection is the ordered bounce."),
    ("What is diffuse reflection?", "Scatter of bounces from a rough surface, no sharp image?", "Paper. Many tiny angles?", "Still light bouncing, not sound?", "Diffuse means no light exists.", "Rough surface, scattered bounces.", "That's it — diffuse is the messy bounce."),
    ("What happens along the normal?", "Ray comes back on itself, both angles zero?", "Special case of the law?", "Retrace?", "It slides along the glass only.", "It retraces along the normal.", "That's it — zero and zero still obey the law."),
    ("Why use a normal at all?", "So both angles are measured the same way?", "Perpendicular reference. Fair compare?", "School diagrams?", "The normal is extra ink only.", "It's the shared reference line.", "That's it — we measure from the normal."),
    ("Does crumpled foil break the law?", "Each tiny smooth bit still obeys locally?", "Many mini-mirrors. Local law?", "Macro looks messy?", "The law fails if foil wrinkles.", "It still holds on each small patch.", "That's it — local law on each patch."),
    ("What is an image in a plane mirror, school-level?", "A virtual image as far behind as the object is in front?", "Equal distance. Left-right story later?", "Stay on reflection.", "The image is a real person stuck in glass.", "A virtual copy behind the mirror.", "That's it — plane-mirror image is virtual, equal distance."),
])

add("Myopia (short-sightedness)", [
    ("What is myopia?", "Blur for distant things because the image forms in front of the retina?", "Blackboard vs book. Far blur?", "Stay on the eye, not on sound.", "Myopia means they cannot think.", "Short sight: far focus too early.", "That's it — myopia is that far-blur geometry."),
    ("What is the retina's job here?", "The screen where a sharp image should sit?", "Focus on the screen. Retina?", "Not a type of lens?", "The retina is a type of battery.", "Where the image should be.", "That's it — retina is the screen."),
    ("What kind of lens corrects myopia?", "A diverging / concave lens?", "Spread rays a little. Concave?", "Not a magnifying convex as the usual fix?", "Any random glass works the same.", "A concave / diverging lens.", "That's it — diverging lens for myopia."),
    ("What might cause the focus to fall short?", "Eyeball too long, or lens too strong?", "Two school causes. Either?", "Geometry of eye?", "Only sadness causes it.", "Too long an eye or too strong a lens.", "That's it — length or extra power."),
    ("What is the near point, loosely?", "Closest you can still focus; myopia people may still read near?", "Near vs far. Different?", "Book ok, board not — pattern?", "Near point is a type of acid.", "The closest comfortable focus.", "That's it — near point is about close focus."),
    ("Is myopia 'weak eyes' as a personality?", "No — it's where the image sits, optics?", "Not intelligence. Optics?", "A focusing-place problem?", "Myopia means they are lazy.", "It's an optics placement problem.", "That's it — not a personality label."),
    ("Why sitting closer is coping not curing?", "It doesn't put far rays on the retina for the board across the room?", "Glasses correct many distances. Closer is a workaround?", "Stay on vision.", "Sitting closer rewrites the eyeball forever.", "Closer hides the problem; a lens corrects focus.", "That's it — coping vs correcting."),
    ("What does a concave lens do to rays, simply?", "Spreads them (diverges) before the eye?", "So the eye's extra power doesn't focus too soon?", "Diverging?", "Concave lenses glue rays together always.", "It diverges the rays a little.", "That's it — concave diverges."),
    ("What is short-sightedness another name for?", "Myopia?", "Same condition. Two names?", "Far blur?", "It means they cannot hear.", "Myopia.", "That's it — same thing, two names."),
    ("Can lens power alone cause myopia-like focus?", "Yes if the lens bends too much?", "Second cause besides long eyeball?", "Too strong a lens?", "Only eyeball length ever matters.", "Yes — too much lens power.", "That's it — power can also put focus early."),
])
