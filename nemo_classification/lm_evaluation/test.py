import requests

if __name__=="__main__":
    res = requests.post(
        "https://api.ai21.com/studio/v1/j1-jumbo/complete",
        headers={"Authorization": "Bearer 8cE9rHDzzriPyvxwsFECqdYRJvqufUNL"},
        json={
            "prompt": "He killed her and said",
            "numResults": 2,
            "maxTokens": 32,
            "stopSequences": ["."],
            "topKReturn": 2,
            "temperature": 0.0
        }
    )
    data = res.json()
    prompt_data = data['prompt']
    text = prompt_data['text']
    tokens = [{
        "token": token_data['generatedToken']['token'],
        "offset": token_data['textRange']['start'],
        "logprob": token_data['generatedToken']['logprob'],
    } for token_data in prompt_data['tokens']]
    print(text)
    print(tokens)