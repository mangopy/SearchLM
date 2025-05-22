from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from transformers import PreTrainedTokenizer

logger = get_logger(__name__)

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


# def _encode_supervised_example(
#         tokenizer: "PreTrainedTokenizer",
#         prompt: Union[Dict[str, str], str],
#         response: Union[Dict[str, str], str],
#         system: Optional[str],
#         cutoff_len: int,
# ) -> Tuple[List[int], List[int]]:
#     messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
#     input_ids = tokenizer.apply_chat_template(conversation=messages)
#     mask_len = len(tokenizer.apply_chat_template(conversation=[messages[0]]))
#     labels = [IGNORE_INDEX] * mask_len + input_ids[mask_len:]
#     return input_ids, labels

def _encode_supervised_example(
        tokenizer: "PreTrainedTokenizer",
        messages: List,
        cutoff_len: int,
        leave_last_one: bool = False
) -> Tuple[List[int], List[int]]:
    input_ids = []
    labels = []
    system_prompt = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": ""}])
    for i in range(0, len(messages), 2):
        user = tokenizer.apply_chat_template(conversation=[messages[i]])
        assistant = tokenizer.apply_chat_template(conversation=[messages[i], messages[i+1]])[len(user):]
        if i!=0:
            user = user[len(system_prompt):]
        input_ids += user + assistant
        if leave_last_one:
            if i+1==len(messages)-1:
                labels += [-100] * len(user) + assistant
            else:
                labels += [-100] * len(user + assistant) 
        else:
            labels += [-100] * len(user) + assistant
    return input_ids[:cutoff_len], labels[:cutoff_len]


def preprocess_supervised_dataset(
        examples: Dict[str, List[Any]],
        tokenizer: "PreTrainedTokenizer",
        cutoff_len: int,
        leave_last_one: bool = False
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    pad_features = ['input_ids', 'labels', 'attention_mask']
    model_inputs = {k:[] for k in pad_features}
    for i in range(len(examples["messages"])):
        # print(examples["messages"][i])
        input_ids, labels = _encode_supervised_example(
            tokenizer=tokenizer,
            messages=examples["messages"][i],
            cutoff_len=cutoff_len,
            leave_last_one = leave_last_one
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    # retain the filed besides the `pad_feature` but only the int or float
    model_inputs.update({k: v for k,v in examples.items() if k in ['prob']})
    return model_inputs

"""
<QUERY> When was "Arthur's Magazine" first published?
<SEARCH> [1] Title: Arthur's Lady's Home Magazine. Content: Arthur's Lady's Home Magazine Arthur's Home Magazine (1852-ca.1898) or Ladies' Home Magazine was an American periodical published in Philadelphia by Timothy Shay Arthur. Editors Arthur and Virginia Francis Townsend selected writing and illustrations intended to appeal to female readers. Among the contributors: Mary Tyler Peabody Mann and Kate Sutherland. In its early years the monthly comprised a selection of articles originally published in Arthur's weekly Home Gazette. Its nonfiction stories contained occasional factual inaccuracies for the sake of a good read. A contemporary review judged it gotten up in good taste and well; and is in nothing overdone. Even its
<FACT> 1844
<QUERY> When was "First for Women" first published?
<SEARCH> [6] Title: Bauer Media Group. Content: several distinct consumer segments: celebrity/entertainment, women's, teen and science/technology. In 1989, the company introduced its second publication, First for Women, a women's magazine. Alliance for Audited Media reports that Woman's World and First for Women are the #1 and #2 selling magazines at retail, respectively. The company's popular teen brands include Twist, launched in 1997; J-14, launched in 1999; M, launched in 2000; Girls' World; launched in 2013, and Animal Tales, launched in 2014. J-14 ranks in the top five media brands for social media presence among all publishers according to Shareablee, a social media research company. In Touch Weekly,
<FACT> 1989
<Final> Arthur's Magazine

USER: ...
ASSISTANT: <QUERY> When was "Arthur's Magazine" first published?
USER: <SEARCH>
ASSISTANT: <FACT> 1844 <QUERY> When was "First for Women" first published?
USER: <SEARCH> 
ASSISTANT: <FACT> 1989 <Final>
USER: Based on the following search results, please answer the question: {question}
ASSISTANT: Arthur's Magazine
"""