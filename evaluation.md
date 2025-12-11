###Evaluation

We evaluated three systems: (1) the baseline RAG pipeline, (2) an improved RAG system with enhanced evidence handling, and (3) a GPT-only model without retrieval. All systems were tested on the same set of veterinary clinical queries. Performance was assessed along three dimensions: correctness, hallucination, and evidence relevance, each scored on a 0–10 scale.



##Correctness

From the results, the GPT-only model ended up with the highest correctness score.
This wasn’t too surprising because the model has seen a lot of medical text during pretraining, so sometimes it can “guess right” even without looking at any reference documents.

The interesting part is that the improved RAG version still performed better than the baseline.
The gain is not huge, but it is consistent.
This tells me that the improvements I added around query handling and retrieval didn’t hurt the model’s ability to answer the question; if anything, they helped it stay on the right track more often.


##Hallucination

Hallucination is where the differences between the systems show up the most.
GPT-only has the highest hallucination by far.
It tends to write confident statements that sound reasonable but aren’t actually supported by anything.

Both RAG systems reduce hallucination a lot, and the improved version is slightly better than the baseline.
The scores aren’t zero, though.
Even with RAG, the model sometimes fills in details that aren’t directly in the retrieved chunk, especially when the evidence is incomplete.
So this result actually makes sense—RAG helps, but it doesn’t “force” the model to stay 100% grounded.


##Evidence Relevance

For evidence relevance, the difference between the baseline and improved system is not very big.
The improved version retrieves slightly better-matched evidence on average, but the improvement is more subtle compared to the other metrics.

However, I noticed that when relevance improves even a little, hallucination usually goes down as well.
So even though the gain looks small numerically, it still helps the answer feel more grounded, and the model makes fewer unsupported claims.



##Overall Comparison

The radar chart puts all three metrics together and makes the trade-offs easier to see.
GPT-only looks good on correctness, but the hallucination score makes it unreliable for anything medical.
Baseline RAG is safer but not always accurate.
The improved RAG system sits somewhere in the middle in a good way: correctness goes up, hallucination goes down, and evidence relevance is slightly better.

For this kind of veterinary question-answering setup, this balance is more realistic than just chasing correctness alone.

#Summary

Overall, the improved RAG system behaved closer to what we’d want in a clinical support tool.
It’s not perfect, but it reduces the risky parts (hallucination) without sacrificing correctness.
The evaluation also shows that retrieval quality matters—even small changes in evidence can affect the model’s final answer.
