# Data Resource

## Training example for Warmup learning
```json
{
    "messages": [
        {
            "role": "user",
            "content": "You are an intelligent search agent capable of simulating a question-answering process by actively seeking information from Wikipedia to answer a given question.\n\nSpecifically, given an open-domain query, please iteratively: (1) Formulate a sub-query to search on Wikipedia; (2) Select useful documents from the search results and (3) Extract supporting facts from the selected documents.\nYour output should include three types of special actions corresponding to the above steps:\n(1) <THINK>: Formulate a sub-query.\n(2) <SEARCH>: Retrieve and carefully read the documents using the formulated sub-query.\n(3) <FACT>: Extract the answer to the sub-query from the documents.\n\nSince this is a multi-hop question, your output should interleave <THINK>, <SEARCH>, and <FACT> actions until reaching the final answer. Conclude your output with the special token <FINAL> followed by the final answer.\n\nBelow is the task for you to complete:\n\n<USER QUERY> What Category 5 hurricane cause the Sqannanoa River to flood?\nYour Output:"
        },
        {
            "role": "assistant",
            "content": "<QUERY> Which hurricanes caused the Swannanoa River to flood?"
        },
        {
            "role": "user",
            "content": "<SEARCH> [0] Title: Swannanoa River. Content: the Asheville metropolitan area, and numerous recreational opportunities. In September 2004, massive rains from the remnants of Hurricane Frances and Hurricane Ivan caused the Swannanoa to flood, causing major damage to the Biltmore Village section of Asheville, and to the other communities that it flows through. Swannanoa River The Swannanoa River flows through the Swannanoa Valley of western North Carolina, and is a major tributary to the French Broad River. It begins at its headwaters in Black Mountain, NC, however, it also has a major tributary near its headwaters: Flat Creek, which begins on the slopes of Mount Mitchell. The"
        },
        {
            "role": "assistant",
            "content": "<FACT> Hurricane Frances and Hurricane Ivan\n<QUERY> Which of these hurricanes was a Category 5 hurricane?"
        },
        {
            "role": "user",
            "content": "<SEARCH> [1] Title: Hurricane Allen. Content: Andrew when the list of names was reused during the 1986 Atlantic hurricane season. When Allen reached Category 5 intensity on August 5, it became the earliest Category 5 storm ever recorded. This record stood until Hurricane Emily surpassed it on July 16, 2005. Allen is one of three Atlantic hurricanes to reach Category 5 on the Saffir-Simpson hurricane scale on three separate occasions, the others being Hurricane Ivan and Hurricane Isabel. Allen also produced the fifth-lowest minimum pressure ever recorded in the Atlantic basin at 899 mbar (hPa) and was the strongest known hurricane in the basin, in terms"
        },
        {
            "role": "assistant",
            "content": "<FACT> Hurricane Ivan\n<FINAL>"
        },
        {
            "role": "user",
            "content": "<SEARCH> end search. please give the final answer to the input question: {question}"
        },
        {
            "role": "assistant",
            "content": "Hurricane Ivan"
        }
    ]
}
```

> Due to the conference requirements, `full dataset` and all the checkpoints will be released after the anonymous peer review period.
