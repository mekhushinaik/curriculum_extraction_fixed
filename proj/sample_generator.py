"""
Generate sample curriculum data in the correct format for testing.
This allows you to verify the output format works correctly.
"""

import json

def generate_sample_curriculum(subject: str, grade: str, curriculum: str) -> list:
    """Generate sample curriculum data in the required format."""
    
    sample_data = {
        "Social Science": [
            {
                "curriculum": curriculum,
                "grade": grade,
                "subject": subject,
                "chapters": [
                    {
                        "id": "chap-social-science-tb-1",
                        "chapterNumber": "1",
                        "chapterTitle": "India - Size and Location",
                        "theme": "Geography",
                        "topics": [
                            {
                                "title": "India's Physical Features",
                                "topic_id": "top-social-science-tb-1-1",
                                "subtopics": [
                                    {
                                        "title": "Mountain ranges and their importance",
                                        "sub_topic_id": "sub-social-science-tb-1-1-1"
                                    },
                                    {
                                        "title": "Major river systems of India",
                                        "sub_topic_id": "sub-social-science-ai-1-1-2"
                                    }
                                ]
                            },
                            {
                                "title": "India's Political Boundaries",
                                "topic_id": "top-social-science-tb-1-2",
                                "subtopics": [
                                    {
                                        "title": "International boundaries and borders",
                                        "sub_topic_id": "sub-social-science-tb-1-2-1"
                                    },
                                    {
                                        "title": "States and union territories",
                                        "sub_topic_id": "sub-social-science-ai-1-2-2"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ],
        "Science": [
            {
                "curriculum": curriculum,
                "grade": grade,
                "subject": subject,
                "chapters": [
                    {
                        "id": "chap-science-tb-1",
                        "chapterNumber": "1",
                        "chapterTitle": "Chemical Reactions and Equations",
                        "theme": "Chemistry",
                        "topics": [
                            {
                                "title": "Types of Chemical Reactions",
                                "topic_id": "top-science-tb-1-1",
                                "subtopics": [
                                    {
                                        "title": "Combination reactions",
                                        "sub_topic_id": "sub-science-tb-1-1-1"
                                    },
                                    {
                                        "title": "Decomposition reactions",
                                        "sub_topic_id": "sub-science-ai-1-1-2"
                                    },
                                    {
                                        "title": "Displacement and double displacement reactions",
                                        "sub_topic_id": "sub-science-ai-1-1-3"
                                    }
                                ]
                            },
                            {
                                "title": "Balancing Chemical Equations",
                                "topic_id": "top-science-tb-1-2",
                                "subtopics": [
                                    {
                                        "title": "Law of conservation of mass",
                                        "sub_topic_id": "sub-science-tb-1-2-1"
                                    },
                                    {
                                        "title": "Method for balancing equations",
                                        "sub_topic_id": "sub-science-ai-1-2-2"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    return sample_data.get(subject, sample_data["Science"])


def save_sample_output(subject: str, grade: str, curriculum: str, subject_slug: str):
    """Save sample curriculum data to JSON file."""
    data = generate_sample_curriculum(subject, grade, curriculum)
    
    output_file = f"output_{subject_slug}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {output_file}")
