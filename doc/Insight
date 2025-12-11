Insight

Looking back at all the numbers and plots together, a few things stood out to me that I didn’t notice at the beginning.

One thing I realized is that correctness alone is pretty misleading.
GPT-only scored the highest in correctness, but once I compared that with the hallucination chart, it became obvious that the model was basically guessing well rather than answering responsibly.
If I had only looked at correctness, I probably would have reached the wrong conclusion about which system was “better.”

Another thing I found interesting is how even small improvements in evidence relevance can stabilize the model’s behavior.
The difference between the baseline and improved system in relevance is not huge, but when relevance went up even a little, hallucination almost always went down.
So the relationship between the two feels stronger than I expected—kind of like getting slightly better evidence helps the model avoid making big leaps or inventing clinical details.

I also realized that RAG does not eliminate hallucination, and honestly this was different from what I assumed earlier.
Even when retrieval is correct, the language model sometimes fills in missing steps on its own.
This is especially obvious in medical questions, where the reasoning chain is long and the evidence rarely tells the whole story.
So the role of RAG feels more like “reducing risk” rather than “forcing correctness.”

Finally, the radar chart made the trade-off very clear:
the improved system isn’t the best in any single dimension, but it’s the only one that forms a balanced shape.
And for clinical or veterinary settings, a balanced model is probably more realistic than a model that scores high on one metric but fails hard on another.

Overall, the main insight for me is that grounding is not about making answers smarter; it’s about making them safer.
And even small retrieval improvements can matter more than I thought, especially when correctness and hallucination pull in different directions.
