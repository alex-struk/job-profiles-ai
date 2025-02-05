from jobstore_ai.llm.main import generate_profile


def runMain():
    # TRY LORA - Generate new profile with lora
    title = "Specialist, Creative"
    prompt = f"Profile: \n\n # **Title:** {title} \n\n **Location:** Vancouver, BC V6B 0N8 CA (Primary) \n\n **Job Type:** Regular Full Time \n\n **Salary Range:** $76,071.18 - $86,658.48 annually \n\n **Close Date:** 2/18/2025"
    new_profile = generate_profile(prompt)
    print(new_profile)


if __name__ == "__main__":
    runMain()