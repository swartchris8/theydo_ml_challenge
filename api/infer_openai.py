import os
import openai
from dotenv import load_dotenv
import pathlib

current_dir = pathlib.Path(__file__).parent.parent.resolve()

load_dotenv(current_dir/".env")

openai.api_key = os.getenv("OPENAI_API_KEY")


def openai_summary(text: str) -> str:
    prompt_instruction = {"\n\nWrite a short summary of the above"}
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{text}{prompt_instruction}",
        temperature=1,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )
    summary = response.get("choices", {"text": "FAIL"})[0].get("text", "FAIL")
    return summary

def openai_translate_german_to_english(text: str) -> str:
    prompt_instruction = {"Translate the below text from German to English\n\n"}
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{prompt_instruction}{text}",
        temperature=1,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )
    summary = response.get("choices", {"text": "FAIL"})[0].get("text", "FAIL")
    return summary

def openai_sentiment(text: str) -> str:
    prompt_instruction = {"Return 1 if the below review has a positive sentiment and 0 if it has a negative sentiment \n\n"}
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{prompt_instruction}{text}",
        temperature=1,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )
    summary = response.get("choices", [{"text": "FAIL"}])[0].get("text", "FAIL")
    return summary

summary_example = """WASHINGTON, June 7 (Reuters) - 

Ukrainians abandoned inundated homes on Wednesday as floods crested across the south after the destruction of the dam, with Russia and Ukraine trading blame for the disaster.

The World Bank will support Ukraine by conducting a rapid assessment of damage and needs after Tuesday's destruction of a huge hydroelectric dam on the front lines between Russian and Ukrainian forces, a top bank official said on Wednesday.

Anna Bjerde, the World Bank's managing director for operations, said on Twitter the destruction of the Novo Kakhovka dam had "many very serious consequences for essential service delivery and the broader environment."

Ukrainian Prime Minister Denys Shmyhal, also writing on Twitter, said he spoke with Bjerde about the impact of the dam's collapse, and she assured him the World Bank would carry out a rapid assessment of the damage and needs.

The World Bank will support Ukraine by conducting a rapid assessment of damage and needs after Tuesday's destruction of a huge hydroelectric dam on the front lines between Russian and Ukrainian forces, a top bank official said on Wednesday.

Ukrainians abandoned inundated homes on Wednesday as floods crested across the south after the destruction of the dam, with Russia and Ukraine trading blame for the disaster.

Ukraine said the deluge would leave hundreds of thousands of people without access to drinking water, swamp tens of thousands of hectares of agricultural land and turn at least 500,000 hectares deprived of irrigation into "deserts"."""

def main():
    print("Testing OPENAI")

    print("SENTIMENT")
    print(openai_sentiment("I absolutely love Inception."))
    print(openai_sentiment("The Box is the worst movie ever"))

    print("GER -> ENG")
    print(openai_translate_german_to_english("Das Haus ist wunderbar. Ich arbeite gerne in NYC."))
    
    print("SUMMARY")
    print(openai_summary(summary_example))




if __name__ == "__main__":
    main()


