SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "\"I don't know based on the provided Medium articles data.\" "
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful.\n\n"
    "Style guidance:\n"
    "- Be concise and direct. Match the user's request exactly (e.g., if asked for "
    "  only titles, return only titles).\n"
    "- For list-style questions, return distinct articles (not multiple chunks of "
    "  the same article).\n"
    "- When citing, mention the article title and author.\n"
    "- Speak directly about the article. Do not say \"based on the provided "
    "  passage/context\", \"according to the retrieved text\", or otherwise refer "
    "  to the retrieval mechanism. The reader does not see the context block."
)


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return (
            f"Question: {question}\n\n"
            "Context: (no relevant passages were retrieved)\n\n"
            "Answer using only the context above."
        )

    parts = ["Context passages from Medium articles:\n"]
    for i, c in enumerate(contexts, start=1):
        parts.append(
            f"[{i}] article_id={c['article_id']} | title={c['title']!r} | "
            f"author={c.get('author', '')!r} | url={c.get('url', '')}\n"
            f"{c['chunk']}\n"
        )
    parts.append(f"\nQuestion: {question}\n\nAnswer using only the context above.")
    return "\n".join(parts)
