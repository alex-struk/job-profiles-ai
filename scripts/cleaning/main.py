import pandas as pd
import numpy as np
from pathlib import Path

column_map_data = {
            "accountabilities": [
                "Accountabilities",
                "SPECIFIC ACCOUNTABILITIES",
                "Responsibilities"
            ],
            "skills": [
                "Knowledge, Skills and Abilities",
                
            ],
             "education_requirements": [
                "Education and Experience",
                "Education and Experience Requirements",
                "Education, experience and skills required",
                "Qualifications:Education and Experience",
                "QualificationsEducation and Experience",
                "Job Requirements:Education and Experience",
                "Job RequirementsEducation and Experience Requirements"
            ],
            "qualifications": [
                "Qualifications",
                "QUALIFICATIONS",
                "Job Requirements",
                "JOB REQUIREMENTS",
                "Must have",
                "Qualifications:Education and Experience Requirements1. Preferred Credentials",
                "Qualifications:Education and Experience Requirements",
                "QUALIFICATIONSEducation and Experience Requirements"
            ],
            "provisos": [
                "Provisos",
                "PROVISOS",
                "WILLINGNESS STATEMENTS",
                "Willingness Statement",
                "PROVISOS/WILLINGNESS STATEMENTS"
            ],
            "application": [
                "APPLICATION REQUIREMENTS",
                "HOW TO APPLY & APPLICATION REQUIREMENTS",
                "How to Apply",
                "How to apply",
                "Application instructions",
                "To Apply",
                "Cover Letter: NO",
                "Cover Letter: NO -",
                "Cover Letter: YES",
                "Cover letter",
                "Resume",
                "Resume required: YES",
                "Resume: YES",
                "Resume: YES -",
                "Questionnaire",
                "Questionnaire: YES",
                "Questionnaire: YES -"
            ],
            "summary": [
                "Job Summary",
                "POSITION SUMMARY",
                "Role description",
                "The Role",
                "PURPOSE OF POSITION"
            ],
            "organization": [
                "About the BCLDB",
                "About the Position",
                "About the Role",
                "About the tribunal",
                "About these Positions",
                "About this Position",
                "About this position",
                "About this role",
                "Employer Description",
                "Working for the BC Public Service",
                "ORGANIZATION CHART",
                "The Team"
            ],
            "salary": [
                "Salary",
                "Salary Range",
                "Remuneration and benefits",
                "WHAT WE OFFER"
            ],
            "location": [
                "Location",
                "Work Option"
            ],
            "competencies": [
                "COMPETENCIES",
                "INDIGENOUS RELATIONS BEHAVIOURAL COMPETENCIES"
            ],
            "classification": [
                "Chief Financial Officer Classification",
                "Classification",
                "Position Type",
                "Job Type",
                "Status"
            ],
            "unknown": [
                "1. Preferred Credentials",
                "119191",
                "2. Expanded Credentials",
                "Additional Information",
                "Additional information",
                "Assessment Process",
                "Close Date",
                "Closing Date",
                "Competition",
                "Competition #",
                "Conditions",
                "Department",
                "Diversity & Inclusion",
                "Human Services fields include",
                "Hybrid Work Model",
                "IMPORTANT",
                "IMPORTANT NOTICE TO APPLICANTS",
                "INFORMATION SESSION",
                "Job Description",
                "Junior Business Analyst (IS 21)",
                "MPORTANT NOTICE TO APPLICANTS",
                "Manager, Brand Storytelling",
                "NOTE",
                "Note",
                "PLEASE NOTE",
                "Position",
                "Position #",
                "Posting close date",
                "Preference may be given for",
                "Preference may be given to applicants with",
                "Preference may be given to candidates with",
                "Preferences",
                "Professional Designation: YES",
                "Professional Designation: YES -",
                "Technical Integration Lead",
                "prior to 9 AM February 18, 2025"
            ]
        }
   

def clean_data():
    """Clean and filter summ_ columns based on criteria"""
    try:
        data_path = 'bc_jobs.csv'
            
        df = pd.read_csv(data_path).copy()
        
        # First get ALL summ_ columns
        summ_columns = [col for col in df.columns if col.startswith('summ_')]

        # print('all summ columns: ', summ_columns)
        
       
       # Filter criteria for summ_ columns only
        exclude_prefixes = ('please', 'Resume', 'Questionnaire', 'Cover Letter', '*')
        valid_summ = [
            col for col in summ_columns
            if not any(col.lower().startswith(p.lower()) for p in exclude_prefixes) 
            and len(col) <= 200
        ]
        
        print('valid_summ: ', str(valid_summ))

        # Check for missing mapped columns - ADD summ_ prefix to mapped columns
        mapped_columns = [f"summ_{item}" for sublist in column_map_data.values() for item in sublist]
        print('mapped_columns: ', str(mapped_columns))

        print('Checking:', [col.replace('summ_', '', 1) for col in valid_summ])
        print('Against:', [m.replace('summ_', '', 1) for m in mapped_columns])

        missing_columns = [col for col in valid_summ if col.replace('summ_', '', 1) not in 
                         [m.replace('summ_', '', 1) for m in mapped_columns]]
        
        if missing_columns:
            print(f"Warning: {len(missing_columns)} columns not in mapping:")
            for col in missing_columns:
                print(f"- {col.replace('summ_', '', 1)}")
        
        # Filter out columns not in mapping
        non_summ_columns = [col for col in df.columns if not col.startswith('summ_')]
        mapped_summ_columns = [col for col in valid_summ if col.replace('summ_', '', 1) in 
                            [m.replace('summ_', '', 1) for m in mapped_columns]]
        print(f"Filtered from {len(df.columns)} to {len(non_summ_columns + mapped_summ_columns)} columns")
        df = df[non_summ_columns + mapped_summ_columns]
         

        # Merge columns according to mapping
        conflict_log = []
        
        for category, source_cols in column_map_data.items():
            # Get columns that exist in the dataframe - ADD summ_ prefix
            existing_cols = [f"summ_{col}" for col in source_cols if f"summ_{col}" in df.columns]
            
            if not existing_cols:
                continue
                
            # Merge columns with conflict handling
            def merge_row(row):
                non_empty = [str(row[col]) for col in existing_cols if pd.notna(row[col])]
                
                if len(non_empty) > 1:
                    conflict_log.append({
                        'record': row.name,
                        'category': category,
                        'columns': existing_cols,
                        'values': non_empty
                    })
                
                return ' | '.join(non_empty) if non_empty else np.nan
            
            df.loc[:, category] = df.apply(merge_row, axis=1)
            
            # Drop original columns
            df.drop(columns=existing_cols, inplace=True)
        
        # Log any conflicts found
        if conflict_log:
            print(f"\nConflict warning: {len(conflict_log)} records had merged columns:")
            for log in conflict_log[:3]:  # Show first 3 conflicts
                print(f"Record {log['record']} in '{log['category']}':")
                print(f"Columns: {', '.join(log['columns'])}")
                print(f"Values: {', '.join(log['values'])}\n")
        
        df = df.drop(columns=['unknown','classification', 'summary', 'Job Summary', 'salary'])

        print(df.columns)

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', None)
        # for col in df.select_dtypes(include=[object]):
        #     df[col] = df[col].str.slice(0, 50)[3]

        print(df)

        df.to_csv('out.csv', index=False) 
        return df

    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        raise

def generate_job_profiles_md(df):
    """Generate Markdown profiles from DataFrame rows using all specified columns"""
    profiles = []
    
    # Verify all required columns exist
    required_columns = ['Close Date', 'Job Category', 'Job Type', 'Location', 'Ministry Branch / Division', 
                        'Ministry/Organization', 'Position Classification', 'Posting Title', 'Salary Range', 
                        'Temporary End Date', 'Union', 'Work Options', 'url', 'accountabilities', 'skills', 
                        'education_requirements', 'qualifications', 'provisos', 'application', 'organization', 
                        'location', 'competencies']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Column display mapping and order
    column_map = [
        ('Posting Title', ''),
        ('Location', 'Location'),
        ('Job Type', 'Job Type'),
        ('Salary Range', 'Salary Range'),
        ('Close Date', 'Close Date'),
        ('Temporary End Date', 'Temporary End Date'),
        ('Ministry/Organization', 'Ministry/Organization'),
        ('Ministry Branch / Division', 'Division'),
        ('Position Classification', 'Position Classification'),
        ('Job Category', 'Job Category'),
        ('Union', 'Union'),
        ('Work Options', 'Work Options'),
        ('url', 'Job URL'),
        ('accountabilities', 'Accountabilities'),
        ('skills', 'Skills'),
        ('education_requirements', 'Education Requirements'),
        ('qualifications', 'Qualifications'),
        ('provisos', 'Provisos'),
        ('application', 'Application Process'),
        ('organization', 'Organization'),
        ('competencies', 'Competencies')
    ]

    for _, row in df.iterrows():
        sections = []
        
        if pd.notna(row['Posting Title']):
            title = row['Posting Title'].split('-', 1)[1].strip() if '-' in row['Posting Title'] else row['Posting Title']
            sections.append(f"# **Title:** {title}\n")
        else:
            continue

        for col, display_name in column_map[1:]:
            if pd.notna(row[col]):
                if col == 'url':
                    sections.append(f"**{display_name}:** [Link]({row[col]})")
                else:
                    sections.append(f"**{display_name}:** {row[col]}")

        profiles.append("\n\n".join(sections))
    
    output_path = 'job_profiles.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Job Profiles\n\n")
        f.write("\n\n---\n\n".join(profiles))

    return profiles


if __name__ == "__main__":
    cleaned = clean_data()
    generate_job_profiles_md(cleaned)
    print("\nFinal DataFrame columns:", cleaned.columns.tolist())


         