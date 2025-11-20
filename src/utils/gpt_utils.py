import pandas as pd
import openai
import time
import pandas as pd
from tqdm import tqdm
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


# ── sampling ───────────────────────────────────────────────────────────────────
def sample_tweets(df, n_total=120, strat_cols=None, random_state=42):
    """
    df .......... DataFrame with at least a 'text' column
    n_total ..... total tweets to return
    strat_cols .. list of cols to stratify on (e.g. ["entropy_bin", "human_label_text"])
    """
    if strat_cols:
        # proportional stratified sample
        df_strat = (
            df.groupby(strat_cols, group_keys=False)
              .apply(lambda x: x.sample(max(1, int(len(x)/len(df)*n_total),
                                             ), random_state=random_state))
        )
        if len(df_strat) > n_total: df_strat = df_strat.sample(n_total, random_state=random_state)
        return df_strat
    else:
        return df.sample(min(n_total, len(df)), random_state=random_state)

def sample_cluster_tweets(
    df_cluster,
    total_samples=100,
    stratify_by_entropy=True,
    entropy_col="entropy_bin",
    random_state=42
):
    """
    Sample tweets from a cluster for GPT-based topic modeling.

    Parameters:
    - df_cluster: DataFrame containing one cluster
    - total_samples: total number of tweets to sample
    - stratify_by_entropy: whether to sample across entropy bins
    - entropy_col: column name that defines entropy bins
    - random_state: reproducibility

    Returns:
    - stratified_sample: DataFrame of stratified sample
    - random_sample: DataFrame of randomly sampled tweets
    """
    # --- Random Sampling ---
    random_sample = df_cluster.sample(n=total_samples, random_state=random_state)

    # --- Stratified Sampling ---
    if stratify_by_entropy:
        entropy_bins = df_cluster[entropy_col].unique()
        entropy_bins = [bin for bin in entropy_bins if pd.notna(bin)]  # Remove NaNs
        n_bins = len(entropy_bins)
        bin_sample_size = total_samples // n_bins
        stratified_sample = pd.concat([
            df_cluster[df_cluster[entropy_col] == bin].sample(
                min(bin_sample_size, len(df_cluster[df_cluster[entropy_col] == bin])),
                random_state=random_state
            ) for bin in entropy_bins
        ])
    else:
        stratified_sample = None

    return stratified_sample, random_sample

def preprocess_texts_for_gpt(df_sample, text_col="text"):
    """Clean and return a list of tweets for prompt construction."""
    return (
        df_sample[text_col]
        .fillna("")
        .str.replace("\n", " ")
        .str.strip()
        .tolist()
    )

# ── chunking ───────────────────────────────────────────────────────────────────
def chunk_tweets(items, size=25):
    """Break list of tweets into equal-sized chunks."""
    return [items[i:i+size] for i in range(0, len(items), size)]

# ── prompt templates ───────────────────────────────────────────────────────────
def prompt_topics(chunk):
    tweets = "\n".join(f"- {t}" for t in chunk)
    return (
        "You are an analyst of disaster-related tweets.\n\n"
        f"Here is a set of tweets:\n{tweets}\n\n"
        "1. List the top 3-5 distinct themes you observe (bullet points).\n"
        "2. For each theme, provide a 1-sentence explanation.\n"
        "3. If any tweets do not fit these themes, mention them briefly under 'Misc'."
    )

def prompt_subgroups(chunk):
    tweets = "\n".join(f"- {t}" for t in chunk)
    return (
        "Below are disaster tweets that may still contain sub-groups.\n"
        f"{tweets}\n\n"
        "Cluster these into 2-3 meaningful sub-groups. "
        "Return a JSON array where each item has keys 'label' and 'example_indices'."
    )

def build_prompt_for_topic_modeling(tweet_chunk):
    tweets_formatted = "\n".join([f"- {t}" for t in tweet_chunk])
    prompt = f"""The following tweets are from the same semantic cluster:

{tweets_formatted}

Based on their content, what are the main topics or themes they discuss? List the top 3–5 topics as bullet points with short explanations."""
    return prompt


def build_prompt_for_cluster_label(tweet_chunk):
    tweets_formatted = "\n".join([f"- {t}" for t in tweet_chunk])
    prompt = f"""Here are several tweets from a single cluster of similar content:

{tweets_formatted}

What would be an appropriate descriptive label or title for this cluster? Summarize it in one sentence or phrase."""
    return prompt


def build_prompt_for_subgrouping(tweet_chunk):
    tweets_formatted = "\n".join([f"- {t}" for t in tweet_chunk])
    prompt = f"""Here is a set of tweets that are semantically similar:

{tweets_formatted}

Can you detect if there are 2–3 subgroups within this list? If so, group the tweets and describe each subgroup briefly."""
    return prompt


# ── GPT call ───────────────────────────────────────────────────────────────────
def gpt_call(prompt, model="gpt-4o-mini", temp=0.3, max_tokens=700):
    r = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temp,
            max_tokens=max_tokens,
        )
    return r.choices[0].message.content.strip() # type: ignore

# Initialize the OpenAI client 
client = openai

def query_gpt_save(
    prompts,
    cluster_id,
    prompt_type="topic_modeling",
    model="gpt-4.0-mini",
    output_path="gpt_responses_cluster.csv",
    delay_between_calls=1.5,
):
    """
    Sends prompts to GPT and saves responses with metadata.
    """
    results = []

    for i, prompt in enumerate(tqdm(prompts, desc=f"Querying GPT for cluster {cluster_id}")):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing tweet content."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=500,
            )
            reply = response.choices[0].message.content.strip() # type: ignore
        except Exception as e:
            print(f"Error on prompt {i}: {e}")
            reply = f"[ERROR] {e}"

        results.append({
            "cluster_id": cluster_id,
            "prompt_type": prompt_type,
            "chunk_index": i,
            "prompt": prompt,
            "response": reply
        })

        time.sleep(delay_between_calls)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"✅ Saved results to: {output_path}")
    return df_results



