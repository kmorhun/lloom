import os
import pandas as pd
import numpy as np
import asyncio
from dotenv import load_dotenv

load_dotenv()

from text_lloom.src.text_lloom import workbench as wb

async def main():
    print("Starting lloom demo")
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

    df = pd.read_excel("data/2024_election.xlsx")
    print(df.columns)
    df= df[['commentID', 'recommendations', 'commentBody']]
    print(df.head())
    lloom = wb.lloom(
        df=df,
        text_col="commentBody",
        id_col="commentID",  # Optional

        # # Model specification
        distill_model_name = "gpt-3.5-turbo-0125",
        embed_model_name = "text-embedding-3-small",
        synth_model_name = "gpt-3.5-turbo-0125",
        score_model_name = "gpt-3.5-turbo-0125",
    )

    cur_seed = None  # Optionally replace with string

    await lloom.gen(seed=cur_seed)
    print(lloom.summary())

if __name__ == "__main__":
    asyncio.run(main())
