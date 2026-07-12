AMD Hackathon Judging FAQ and Self-Check Guide
Quick Note
The leaderboard is still catching up. Some errors are infrastructure-related and do not reflect your project quality.
Please do not keep resubmitting to move up the queue. It does not help and it adds more load.
Exact internal judge prompts and hidden test cases are not shared. The scoring criteria are stable: correctness, required format, reliability, runtime, and track-specific quality.
Public Validation Examples

We are sharing a public validation set made from retired scoring examples. These examples are no longer part of final judging.

Use them to check:
- Input/output format.
- Expected answer quality.
- Style expectations.
- Local container behavior.

Final judging will use a separate hidden set with the same format, difficulty, and scoring principles. Passing the public examples is helpful, but it does not guarantee a final score.

Sample Tasks
These are public validation examples from retired scoring cases. They are provided for local testing and are not part of final judging.

Track 1 sample tasks

T01 - factual_knowledge
Prompt: Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB.
Expected: Correctly identifies red, green, and blue as the three primary colors, and explains that RGB is used in displays because screens emit light additively (additive color mixing), whereas RYB applies to subtractive mixing of physical pigments.

T01b - factual_knowledge
Prompt: What is the difference between machine learning and deep learning? Briefly explain how each works.
Expected: Distinguishes ML as algorithms that learn patterns from data (statistical or feature-based), and deep learning as a subset of ML using multi-layer neural networks. Explains that deep learning automatically extracts features from raw data, while traditional ML often requires manual feature engineering. Must make the subset relationship clear.

T01c - factual_knowledge
Prompt: Explain the difference between RAM and ROM in a computer. What is each type used for?
Expected: RAM (Random Access Memory) is volatile and fast, used for temporary storage of active programs and data. ROM (Read-Only Memory) is non-volatile and stores permanent firmware or BIOS. Must correctly distinguish volatility, speed, and use case for both types.

T02 - mathematical_reasoning
Prompt: A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?
Expected: Correctly arrives at 1,672 units remaining. The calculation must follow: 2400 minus 888 (37% of 2400) equals 1512, plus 800 equals 2312, minus 640 equals 1672. Minor arithmetic shown or implied.

T02b - mathematical_reasoning
Prompt: A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?
Expected: Correctly calculates: 30 cookies need (3/4 × 30/12) = 1.875 cups of sugar. Total cost = 1.875 × $2.40 = $4.50. Both answers must be correct. Minor rounding (1.87 or 1.88 cups) is acceptable if the final cost rounds correctly to $4.50.

T03 - sentiment_classification
Prompt: Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'
Expected: Classifies the sentiment as Mixed, Neutral, or Positive (any of these three labels is acceptable), provided the one-sentence reason acknowledges both the negative experience (late delivery, damaged packaging) and the positive outcome (working product, responsive support). A Negative classification does not pass. A reason that acknowledges only one side does not pass, regardless of the label.

T03b - sentiment_classification
Prompt: Classify the sentiment of this tweet as Positive, Negative, or Neutral and give a one-sentence reason: 'Just got my order. Box was dented and the manual was missing, but honestly the device itself is flawless and set up in under 5 minutes.'
Expected: Classifies as Mixed, Neutral, or Positive (any of these three labels is acceptable), provided the reason acknowledges both negative aspects (dented box, missing manual) and positive aspects (flawless device, fast setup). A Negative classification does not pass. A reason that acknowledges only one side does not pass, regardless of the label.

T04 - text_summarization
Prompt: Summarize the following passage in exactly two sentences:

'Machine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records that might be missed by human clinicians. However, concerns remain about model interpretability, data privacy, liability when errors occur, and the potential for algorithmic bias to worsen existing healthcare disparities. Regulatory frameworks are still catching up with the pace of deployment, creating uncertainty for healthcare providers and technology developers alike.'
Expected: Produces exactly two sentences. The summary captures both the opportunity (ML assisting clinical tasks such as image analysis, prediction, and pattern recognition) and the key challenges (interpretability, bias, privacy, liability, and regulatory lag). Omitting either side, or producing more or fewer than two sentences, does not pass.

T04b - text_summarization
Prompt: Summarize the following passage in exactly three bullet points, each no longer than 15 words:

'Remote work has transformed how companies operate globally. Employees gain flexibility and reduced commute times, leading to reported improvements in work-life balance. However, challenges persist around collaboration, company culture, and the blurring of personal and professional boundaries. Organisations are responding by investing in digital collaboration tools and rethinking office space as a hub for social and creative work rather than daily attendance.'
Expected: Produces exactly three bullet points, each under 15 words. Points must cover: (1) remote work benefits such as flexibility and work-life balance, (2) challenges around collaboration, culture, and boundary blur, (3) organisational response through digital tools and reimagined office use. More or fewer than three bullets, or any bullet exceeding 15 words, does not pass.

T05 - named_entity_recognition
Prompt: Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE:

'On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.'
Expected: Correctly identifies and labels all five distinct entities: Sundar Pichai (PERSON), March 15 2023 (DATE), Google (ORGANIZATION), Zurich (LOCATION), ETH Zurich (ORGANIZATION). All five must be present with correct labels. Missing an entity or mislabelling more than one does not pass.

Before You Submit
Pull your exact Docker image tag from a clean machine.
Run the container without local files or manual setup.
Confirm it writes the expected output.
Validate the JSON output.
Return results for every required task.
Stay under the runtime limit.
Make sure your repo or image tag is public and accessible.

Common Errors
PULL_ERROR
The judging system could not pull your Docker image.

Check:
- Image is public.
- Image name and tag are exact.
- Tag exists in the registry.
- It works outside your local Docker cache.

RUNTIME_ERROR
Your container started but crashed.

Check:
- All dependencies are inside the image.
- Entrypoint runs automatically.
- No local-only files are required.
- No private secrets are required.

TIMEOUT
Your container took too long.

Check:
- Avoid downloading large files at runtime.
- Avoid slow setup during evaluation.
- Test the worst-case runtime locally.
- Do not leave processes hanging.

OUTPUT_MISSING
Your container did not create the expected output.

Check:
- Output path matches the spec.
- Output is written before exit.
- Failures still produce a valid response when possible.

INVALID_RESULTS_SCHEMA
Your output exists, but the structure is wrong.

Check:
- JSON is valid.
- Required fields are present.
- Field names match the spec.
- Types match the spec.

MISSING_TASKS
Your output skipped one or more tasks.

Check:
- Return one result per input task.
- Preserve task IDs exactly.
- Do not silently skip failed tasks.

ACCURACY_GATE_FAILED
Your output was valid, but the answer quality did not pass the accuracy threshold.

Check:
- Prioritize correctness first.
- Avoid generic answers.
- Answer the actual task directly.
- For Track 1, token efficiency matters after correctness.

INFRA_ERROR
A backend-side scoring issue occurred.

What to know:
- This does not automatically mean your project failed.
- These cases are handled by the judging process.
- Repeated resubmission is not needed.


Track 1: Token Efficient Routing Agent

Focus on:
- Correct answers.
- Complete task coverage.
- Valid output format.
- Reliable execution.
- Token/cost efficiency after correctness.

Self-check:
- Every task ID has an answer.
- Output JSON is valid.
- Container runs cleanly.
- Runtime is under the limit.
- Local inference is fine if the output is correct.

