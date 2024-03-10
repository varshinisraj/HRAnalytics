import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression




# Read the Excel file and load the data into a DataFrame
df = pd.read_csv('C:/Users/Varshini S Raj/PycharmProjects/HRAnalytics/HR_Analytics.csv')

# Calculate the total number of employees
total_employees = len(df)

# Calculate the total number of male and female employees
male_employees = len(df[df['Gender'] == 'Male'])
female_employees = len(df[df['Gender'] == 'Female'])

# Calculate the percentage of male and female employees
total_employees = male_employees + female_employees  ;
male_percentage = (male_employees / total_employees) * 100
female_percentage = (female_employees / total_employees) * 100

# Create a pie chart using Plotly
labels = ['Male', 'Female']
values = [male_percentage, female_percentage]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

# Calculate the top 5 employees with the highest and lowest total working years
top_highest_employees = df.sort_values('TotalWorkingYears', ascending=False).head(5)[['EmpID', 'Age', 'MonthlyIncome', 'TotalWorkingYears']]
top_lowest_employees = df.sort_values('TotalWorkingYears', ascending=True).head(5)[['EmpID', 'Age', 'MonthlyIncome', 'TotalWorkingYears']]

# Display the results in the Streamlit app

with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        image = Image.open('HR icon.jpeg')
        image = image.resize((140, 140))
        st.image(image)

    with col2:
        st.markdown("<h1 style='color: red; font-family: Calibri;'>HR Analytics</h1>", unsafe_allow_html=True)

    # Sidebar - Search Employee
    st.sidebar.subheader("Search Employee")
    search_empid = st.sidebar.text_input("Enter Employee ID")
    search_button = st.sidebar.button("Search")

    # Sidebar - Download CSV
    st.sidebar.subheader("Download CSV")
    csv_download = st.sidebar.button("Download HR Analytics Data")

    if csv_download:
        # Create a link to download the CSV file
        csv_file = df.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="HR_Analytics.csv">Download CSV File</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)


def attrition_analysis():
    st.markdown("<h2 style='text-align: center; color: red; text-decoration: underline;'>Attrition Analysis</h2>", unsafe_allow_html=True)

    # Calculate attrition rate by AgeGroup
    age_group_attrition = df.groupby(['AgeGroup', 'Attrition'])['EmpID'].count().unstack()
    st.write("<h3 style='color: red;'>Attrition Rate by AgeGroup</h3>", unsafe_allow_html=True)
    st.dataframe(age_group_attrition)

    # Bar chart for attrition rate by AgeGroup
    st.bar_chart(age_group_attrition)

    # Calculate attrition rate by EducationField
    education_field_attrition = df.groupby(['EducationField', 'Attrition'])['EmpID'].count().unstack()
    st.write("<h3 style='color: red;'>Attrition Rate by EducationField</h3>", unsafe_allow_html=True)
    st.dataframe(education_field_attrition)

    # Bar chart for attrition rate by EducationField
    st.bar_chart(education_field_attrition)

    # Calculate attrition rate by Department
    department_attrition = df.groupby(['Department', 'Attrition'])['EmpID'].count().unstack()
    st.write("<h3 style='color: red;'>Attrition Rate by Department</h3>", unsafe_allow_html=True)
    st.dataframe(department_attrition)

    # Bar chart for attrition rate by Department
    st.bar_chart(department_attrition)

    # Calculate overall attrition rate
    overall_attrition_rate = df['Attrition'].value_counts(normalize=True) * 100
    st.write("<h3 style='color: red;'>Overall Attrition Rate</h3>", unsafe_allow_html=True)
    st.dataframe(overall_attrition_rate)

    # Pie chart for overall attrition rate
    labels = overall_attrition_rate.index
    values = overall_attrition_rate.values
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig)

# Select option
selected_option = st.sidebar.radio('Select an option', ('Employee Analysis', 'Age demographics', 'Attrition Analysis', 'Salary Prediction', 'Promotion Prediction'))

if selected_option == 'Employee Analysis':
    # Add spacing
    st.write('\n\n')

    st.markdown("<h2 style='text-align: center; color: red; text-decoration: underline;'>Employee Records</h2>",
                unsafe_allow_html=True)

    # Perform search and display employee details
    if search_button:
        employee = df[df['EmpID'] == search_empid]
        if not employee.empty:
            st.subheader(f"Employee Details for EmpID: {search_empid}")
            st.write(employee)
        else:
            st.sidebar.warning("Employee not found.")

    # Customize the appearance of the text using Markdown syntax
    st.markdown("<p style='font-size: 20px; color: black; font-weight: bold; font-family: Calibri;'>Total Employees: "
                "<span style='font-size: 18px; color: red; font-weight: bold;'>{}</span></p>".format(total_employees),
                unsafe_allow_html=True)

    st.markdown("<p style='font-size: 20px; color: black; font-weight: bold; font-family: Calibri;'>Male Employees: "
                "<span style='font-size: 18px; color: red; font-weight: bold;'>{}</span></p>".format(male_employees),
                unsafe_allow_html=True)

    st.markdown("<p style='font-size: 20px; color: black; font-weight: bold; font-family: Calibri;'>Female Employees: "
                "<span style='font-size: 18px; color: red; font-weight: bold;'>{}</span></p>".format(female_employees),
                unsafe_allow_html=True)

    # Display the pie chart
    st.plotly_chart(fig)

    # Add interactivity with checkbox
    if st.checkbox("Show Employee Statistics"):
        st.subheader("Employee Statistics")
        st.write("Here are some statistics about the employees:")

        # Display the highest and lowest total working years
        st.write(f"- Highest Total Working Years: {df['TotalWorkingYears'].max()}")
        st.write(f"- Lowest Total Working Years: {df['TotalWorkingYears'].min()}")

        # Display the total number of employees with highest and lowest total working years
        highest_total_employees = len(df[df['TotalWorkingYears'] == df['TotalWorkingYears'].max()])
        lowest_total_employees = len(df[df['TotalWorkingYears'] == df['TotalWorkingYears'].min()])
        st.write(f"- Total Employees with Highest Total Working Years: {highest_total_employees}")
        st.write(f"- Total Employees with Lowest Total Working Years: {lowest_total_employees}")

        # Display the total number of female employees with highest and lowest total working years
        female_highest_total_employees = len(
            df[(df['Gender'] == 'Female') & (df['TotalWorkingYears'] == df[df['Gender'] == 'Female']['TotalWorkingYears'].max())])
        female_lowest_total_employees = len(
            df[(df['Gender'] == 'Female') & (df['TotalWorkingYears'] == df[df['Gender'] == 'Female']['TotalWorkingYears'].min())])
        st.write(f"- Female Employees with Highest Total Working Years: {female_highest_total_employees}")
        st.write(f"- Female Employees with Lowest Total Working Years: {female_lowest_total_employees}")

        # Display the total number of male employees with highest and lowest total working years
        male_highest_total_employees = len(
            df[(df['Gender'] == 'Male') & (df['TotalWorkingYears'] == df[df['Gender'] == 'Male']['TotalWorkingYears'].max())])
        male_lowest_total_employees = len(
            df[(df['Gender'] == 'Male') & (df['TotalWorkingYears'] == df[df['Gender'] == 'Male']['TotalWorkingYears'].min())])
        st.write(f"- Male Employees with Highest Total Working Years: {male_highest_total_employees}")
        st.write(f"- Male Employees with Lowest Total Working Years: {male_lowest_total_employees}")

    # Display the top 5 employees with the highest and lowest total working years
    st.markdown("<h3 style='color: red;'>Top Employees with Highest Total Working Years</h3>",
                 unsafe_allow_html=True)
    st.write(top_highest_employees)

    st.markdown("<h3 style='color: red;'>Top Employees with Lowest Total Working Years</h3>",
                 unsafe_allow_html=True)
    st.write(top_lowest_employees)

#Age demographics

if selected_option == 'Age demographics':
    st.markdown("<h2 style='text-align: center; color: red; text-decoration: underline;'>Age Demographics</h2>", unsafe_allow_html=True)

    # Calculate the age distribution
    age_distribution = df['Age'].value_counts().sort_index().reset_index()
    age_distribution.columns = ['Age', 'Count']

    # Calculate the average age
    average_age = df['Age'].mean()

    # Calculate the age range
    min_age = df['Age'].min()
    max_age = df['Age'].max()

    # Display the age distribution
    st.markdown("<h3 style='color: red;'>Age Distribution</h3>", unsafe_allow_html=True)
    st.dataframe(age_distribution.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    # Display the average age
    st.markdown("<span style='color: red;'>Average Age:</span> {:.1f}".format(average_age), unsafe_allow_html=True)

    # Display the age range
    st.markdown("<span style='color: red;'>Age Range:</span> {} to {}".format(min_age, max_age), unsafe_allow_html=True)

    # Calculate the age distribution by gender
    age_by_gender = df.groupby('Gender')['Age'].value_counts().unstack()

    # Display the age distribution by gender
    st.markdown("<h3 style='color: red;'>Age Distribution by Gender</h3>", unsafe_allow_html=True)
    st.dataframe(age_by_gender)
    st.bar_chart(age_by_gender)

    # Calculate the age distribution by department
    age_by_department = df.groupby('Department')['Age'].value_counts().unstack()

    # Display the age distribution by department
    st.markdown("<h3 style='color: red;'>Age Distribution by Department</h3>", unsafe_allow_html=True)
    st.dataframe(age_by_department)
    st.bar_chart(age_by_department)

    # Calculate the age distribution by job role
    age_by_job_role = df.groupby('JobRole')['Age'].value_counts().unstack()

    # Display the age distribution by job role
    st.markdown("<h3 style='color: red;'>Age Distribution by Job Role</h3>", unsafe_allow_html=True)
    st.dataframe(age_by_job_role)
    st.bar_chart(age_by_job_role)

    # Calculate the age distribution by attrition
    age_by_attrition = df.groupby('Attrition')['Age'].value_counts().unstack()

    # Display the age distribution by attrition
    st.markdown("<h3 style='color: red;'>Age Distribution by Attrition</h3>", unsafe_allow_html=True)
    st.dataframe(age_by_attrition)
    st.bar_chart(age_by_attrition)

    # Calculate the age distribution by performance rating
    age_by_performance = df.groupby('PerformanceRating')['Age'].value_counts().unstack()

    # Display the age distribution by performance rating
    st.markdown("<h3 style='color: red;'>Age Distribution by Performance Rating</h3>", unsafe_allow_html=True)
    st.dataframe(age_by_performance)
    st.bar_chart(age_by_performance)

    # Calculate the age distribution by tenure
    age_by_tenure = df.groupby('YearsAtCompany')['Age'].value_counts().unstack()

    # Display the age distribution by tenure
    st.markdown("<h3 style='color: red;'>Age Distribution by Tenure</h3>", unsafe_allow_html=True)
    st.dataframe(age_by_tenure)
    st.bar_chart(age_by_tenure)

# Attrition analysis
if selected_option == 'Attrition Analysis':
    attrition_analysis()


# Customize theme and styling
# Set the background color of the main content
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f7;
    }
    </style>
    """,
    unsafe_allow_html=True
)



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Salary Prediction
if selected_option == 'Salary Prediction':
    st.markdown("<h2 style='text-align: center; color: red; text-decoration: underline;'>Salary Prediction</h2>",
                unsafe_allow_html=True)

    st.write("Please provide employee information for prediction:")
    performance_rating = st.number_input("Performance Rating", min_value=1, max_value=5)
    job_satisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=4)
    years_at_company = st.number_input("Years at Company", min_value=0)

    # Add a dropdown for Education Field
    education_field = st.selectbox("Education Field", df['EducationField'].unique())

    # Create a button to make the prediction
    predict_button = st.button("Predict Salary")

    if predict_button:
        # Convert categorical features to numerical using label encoding
        df_encoded = df.copy()
        le = LabelEncoder()
        df_encoded['EducationField'] = le.fit_transform(df_encoded['EducationField'])

        # Define features and target variable
        features = ['PerformanceRating', 'JobSatisfaction', 'YearsAtCompany', 'EducationField']
        target = 'MonthlyIncome'

        # Split the data into training and testing sets
        train_data, test_data, train_target, test_target = train_test_split(
            df_encoded[features], df_encoded[target], test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(train_data, train_target)

        # Prepare input data for prediction
        education_field_encoded = le.transform([education_field])[0]
        input_data = [[performance_rating, job_satisfaction, years_at_company, education_field_encoded]]

        # Make predictions
        prediction = model.predict(input_data)[0]

        # Display prediction result with 2 decimal places
        st.write("Predicted Salary:", round(prediction, 2))



# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import streamlit as st

# ... (Read and preprocess your data here)

# Promotion Prediction
if selected_option == 'Promotion Prediction':
    st.markdown("<h2 style='text-align: center; color: red; text-decoration: underline;'>Promotion Prediction</h2>",
                unsafe_allow_html=True)

    st.write("Please provide employee information for promotion prediction:")
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0)
    job_involvement = st.number_input("Job Involvement", min_value=1, max_value=4)
    performance_rating = st.number_input("Performance Rating", min_value=1, max_value=5)

    # Create a button to make the prediction
    predict_promotion_button = st.button("Predict Promotion")

    if predict_promotion_button:
        # Convert categorical features to numerical using label encoding
        df_encoded = df.copy()
        le = LabelEncoder()
        df_encoded['JobRole'] = le.fit_transform(df_encoded['JobRole'])

        # Prepare data for clustering
        features = ['YearsSinceLastPromotion', 'JobInvolvement', 'PerformanceRating', 'JobRole']
        clustering_data = df_encoded[features]

        # Apply KMeans clustering
        num_clusters = 2  # You can adjust this based on your needs
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(clustering_data)

        # Add cluster information back to the DataFrame
        df_encoded['Cluster'] = clusters

        # Identify potential promotion candidates
        potential_candidates = df_encoded[
            (df_encoded['YearsSinceLastPromotion'] <= years_since_last_promotion) &
            (df_encoded['JobInvolvement'] >= job_involvement) &
            (df_encoded['Attrition'] == 'No')
        ]

        # Display potential candidates' EmpID in a comma-separated format
        potential_candidate_empids = ', '.join(map(str, potential_candidates['EmpID'].tolist()))
        st.write("Potential Promotion Candidates (EmpID):", potential_candidate_empids)

        # Display total number of eligible employees for promotion
        total_eligible_employees = len(potential_candidates)
        st.write("Total Eligible Employees for Promotion:", total_eligible_employees)


