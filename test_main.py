import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile
import pycountry
import os
from main import CreateDashboard, GrabAirTableData
import io

# Must be the first Streamlit command
st.set_page_config(page_title="Immigration Case Dashboard", layout="wide")

class TestDashboard(CreateDashboard):
    def __init__(self):
        # Load test data instead of Airtable data
        self.airtable_data = self.load_test_data()
        if not self.airtable_data.empty:
            # Clean and prepare the data before passing to parent
            self.airtable_data = self.prepare_data(self.airtable_data)
            
            # Extra step to ensure date columns are compatible with parent class
            # Replace "Unknown" with NaT for date columns that the parent will try to convert
            critical_date_cols = ['Client Date of Birth', 'Case Priority Date', 'Visa Expiry Date']
            for col in critical_date_cols:
                if col in self.airtable_data.columns:
                    # For string columns with 'Unknown' values
                    if self.airtable_data[col].dtype == 'object':
                        # Replace 'Unknown' with empty string so parent's to_datetime will create NaT
                        self.airtable_data[col] = self.airtable_data[col].replace('Unknown', '')
            
        # Skip parent class initialization with super() to avoid date conversion issues
        # super().__init__(self.airtable_data)
        
        # Instead, manually initialize required attributes from the parent class
        self.check_dates_match = None
        self.check_dates_no_match = None
        self.create_warning_charts = None
        
        # Other initialization that might be needed from parent class
        self.initialize_dashboard_data()

    def initialize_dashboard_data(self):
        """Initialize dashboard data without calling parent class methods that cause date conversion issues"""
        try:
            # Set additional attributes that would be created in the parent class
            # Example: Calculate ages for valid date of birth entries
            if 'Client Date of Birth' in self.airtable_data.columns and 'Age' not in self.airtable_data.columns:
                try:
                    # Only calculate ages for valid date strings
                    valid_dob = self.airtable_data['Client Date of Birth'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)
                    self.airtable_data['Age'] = None
                    
                    if valid_dob.any():
                        # Convert valid dates to datetime for calculation
                        temp_dates = pd.to_datetime(self.airtable_data.loc[valid_dob, 'Client Date of Birth'], errors='coerce')
                        today = pd.Timestamp.today()
                        
                        # Calculate ages where dates are valid
                        ages = (today - temp_dates).dt.days / 365.25
                        self.airtable_data.loc[valid_dob, 'Age'] = ages.round(1)
                except Exception as e:
                    st.sidebar.warning(f"Could not calculate ages: {e}")

            # Initialize any other dashboard data needed
        except Exception as e:
            st.error(f"Error initializing dashboard data: {e}")

    def calculate_age(self, birth_date):
        """Calculate age based on birth date"""
        if not birth_date or birth_date == '' or birth_date == 'Unknown':
            return 0
            
        try:
            if isinstance(birth_date, str):
                birth_date = pd.to_datetime(birth_date, errors='coerce')
                
            if pd.isnull(birth_date):
                return 0
                
            today = pd.Timestamp.today()
            age = (today - birth_date).days / 365.25
            return round(age, 1)
        except Exception:
            return 0
    
    def get_age_group(self, age):
        """Convert age to age group category"""
        # Define valid age groups
        valid_groups = ['Unknown', 'Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        
        # Determine the correct group
        if pd.isnull(age) or age <= 0:
            group = 'Unknown'
        elif age < 18:
            group = 'Under 18'
        elif age < 25:
            group = '18-24'
        elif age < 35:
            group = '25-34'
        elif age < 45:
            group = '35-44'
        elif age < 55:
            group = '45-54'
        elif age < 65:
            group = '55-64'
        else:
            group = '65+'
            
        # Ensure the group is in our valid list
        if group not in valid_groups:
            group = 'Unknown'
            
        return group

    def prepare_data(self, df):
        """Clean and prepare the data before processing."""
        try:
            # Create a clean copy to avoid fragmentation
            df = df.copy()
            
            # Skip showing data info for better performance
            # st.sidebar.write("Data Info:")
            # buffer = io.StringIO()
            # df.info(buf=buffer)
            # st.sidebar.text(buffer.getvalue())
            
            # Skip showing rows for better performance 
            # st.sidebar.write("First few rows:")
            # st.sidebar.write(df.head())

            # Define date formats for different types of columns
            date_formats = {
                'Created Time': '%Y-%m-%d %H:%M:%S',  # For timestamps
                'default': '%Y-%m-%d'  # For regular dates
            }
            
            # Define date columns and their expected formats
            date_columns = {
                'Created Time': date_formats['Created Time'],  # Special format for timestamps
                'Client Date of Birth': date_formats['default'],
                'Case Priority Date': date_formats['default'],
                'Case Open Date': date_formats['default'],
                'Case Approved On': date_formats['default'],
                'Approval Valid From': date_formats['default'],
                'Approval Valid Till': date_formats['default'],
                'Case Active/Inactive Date': date_formats['default'],
                'Case Receipt Date': date_formats['default'],
                'RFE Received On': date_formats['default'],
                'RFE Due On': date_formats['default'],
                'RFE Submitted On': date_formats['default'],
                'Case Denied On': date_formats['default'],
                'Case Closed On': date_formats['default'],
                'Client Created Date': date_formats['default'],
                'Sent to Govt Agency On': date_formats['default'],
                'I-94 Valid From': date_formats['default'],
                'I-94 Valid to': date_formats['default'],
                'I-797 Valid From': date_formats['default'],
                'I-797 valid to': date_formats['default'],
                'APD Valid From': date_formats['default'],
                'APD Valid To': date_formats['default'],
                'EAD/AP Valid From': date_formats['default'],
                'EAD/AP Expiration Date': date_formats['default'],
                'EAD Valid To': date_formats['default'],
                'EAD Valid From': date_formats['default'],
                'Current Immigration Status Valid From': date_formats['default'],
                'Current Immigration Status Valid To': date_formats['default'],
                'H Or L Status Valid From': date_formats['default'],
                'H or L Maxout Date': date_formats['default'],
                'Priority Date-1 - Priority Date': date_formats['default'],
                'Priority Date-2 - Priority Date': date_formats['default'],
                'Visa Expiry Date': date_formats['default'],
                'I-485 Filing Date': date_formats['default'],
                'Green Card Valid To': date_formats['default'],
                'PERM Eligibility Date': date_formats['default'],
                'Last Step Completed Date': date_formats['default'],
                'Next Step To Be Completed Date': date_formats['default'],
                'Last Action Due Date': date_formats['default'],
                'Next Action Due Date': date_formats['default']
            }

            # Initialize the transformed data dictionary
            transformed_data = {}

            # Process date columns - Convert ALL to strings for better compatibility
            for col, date_format in date_columns.items():
                if col in df.columns:
                    try:
                        # First try with the specified format
                        temp_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                        if temp_dates.isna().all():
                            # If all NaT, try without format specification
                            temp_dates = pd.to_datetime(df[col], errors='coerce')
                        
                        # Skip datetime storing and convert directly to strings 
                        # This avoids Arrow conversion issues later
                        transformed_data[col] = temp_dates.dt.strftime(date_formats.get(col, date_formats['default'])).fillna('')
                        
                    except Exception as e:
                        st.sidebar.warning(f"Error converting dates in {col}: {str(e)}")
                        # Fallback to empty strings for problematic date columns
                        transformed_data[col] = df[col].astype(str).replace({'nan': '', 'NaT': '', 'None': '', 'Unknown': ''})

            # Process numeric columns with specific dtypes
            numeric_columns = {
                'Annual Salary': 'float64',
                'Number of Days Left for H or L': 'int64'  # Standard int64
            }

            for col, dtype in numeric_columns.items():
                if col in df.columns:
                    try:
                        # Convert to numeric, coerce errors, fill NA with 0
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        transformed_data[col] = numeric_series.fillna(0).astype(dtype)
                    except Exception as e:
                        st.sidebar.warning(f"Error converting numeric column {col}: {str(e)}")
                        # Fill with 0 for all numeric columns
                        transformed_data[col] = 0

            # Process categorical columns
            categorical_columns = ['Case Status', 'Case Type', 'Processing Type', 'Corporation Type',
                                'Service Center', 'Corporation Status', 'Is Case Active?']

            for col in categorical_columns:
                if col in df.columns:
                    try:
                        # Convert to string first, fill NA, then create category
                        # This adds 'Unknown' to the category list from the beginning
                        values = df[col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'}).fillna('Unknown')
                        
                        # Get unique categories including 'Unknown'
                        unique_values = values.unique().tolist()
                        if 'Unknown' not in unique_values:
                            unique_values.append('Unknown')
                            
                        # Create categorical with predefined categories
                        transformed_data[col] = pd.Categorical(values, categories=unique_values)
                    except Exception as e:
                        st.sidebar.warning(f"Error converting categorical column {col}: {str(e)}")
                        # Fall back to plain strings without categories
                        transformed_data[col] = df[col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'}).fillna('Unknown')

            # Handle remaining columns as strings, ensuring NA handling
            remaining_columns = [col for col in df.columns if col not in transformed_data]
            for col in remaining_columns:
                 # Fill NA with a placeholder string, then convert to string type
                transformed_data[col] = df[col].fillna('Unknown').astype(str)

            # Create new dataframe with transformed data
            df_transformed = pd.DataFrame(transformed_data, index=df.index)
            
            # No need for additional conversion since we already transformed to strings
            
            # Reapply original column order and add any missing columns back (filled with 'Unknown')
            df_transformed = df_transformed.reindex(columns=df.columns)
            for col in df.columns:
                 if col not in df_transformed:
                     df_transformed[col] = 'Unknown'

            # Show basic info about the prepared data
            st.sidebar.write(f"Prepared {len(df_transformed)} rows with {len(df_transformed.columns)} columns")

            return df_transformed

        except Exception as e:
            st.sidebar.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame(columns=df.columns)  # Return empty DataFrame with same structure

    def load_test_data(self):
        try:
            # First, try to load the Case Status Report file
            default_file = 'Case Status Report (Apr-14-2025 15-59-02).xlsx'
            
            # Show file upload option
            uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx'])
            
            if uploaded_file is not None:
                try:
                    # Read Excel with openpyxl engine and suppress the style warning
                    import warnings
                    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                    st.sidebar.success("Excel file loaded successfully!")
                    st.sidebar.write("Columns found:", df.columns.tolist())
                    
                    # Prepare data for Arrow compatibility
                    df = self.prepare_data(df)
                    return df
                except Exception as e:
                    st.sidebar.error(f"Error reading uploaded file: {str(e)}")
                    return pd.DataFrame()
            
            # If no upload, try default file
            if os.path.exists(default_file):
                try:
                    # Read Excel with openpyxl engine and suppress the style warning
                    import warnings
                    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
                    df = pd.read_excel(default_file, engine='openpyxl')
                    st.sidebar.info(f"Loaded Case Status Report file")
                    st.sidebar.write("Columns found:", df.columns.tolist())
                    
                    # Prepare data for Arrow compatibility
                    df = self.prepare_data(df)
                    return df
                except Exception as e:
                    st.sidebar.error(f"Error reading Case Status Report: {str(e)}")
                    return pd.DataFrame()
            else:
                st.sidebar.warning("Case Status Report file not found in the current directory.")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return pd.DataFrame()

    def run_dashboard(self):
        # Add dashboard header
        st.markdown(
            """
            <div style='background-color: #ffeb3b; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
                <h2 style='color: #000000; margin: 0;'>📊 Case Status Dashboard</h2>
                <p style='color: #000000; margin: 5px 0 0 0;'>Using Case Status Report data. You can also upload a different Excel file using the sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if self.airtable_data.empty:
            st.warning("No data available. Please check if the Case Status Report file exists and has the correct format.")
            return

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Case Analysis", 
            "Age Demographics", 
            "Additional Demographics",
            "Worldwide Distribution",
            "Case Status Insights",
            "Corporate Overview"
            # "🧪 Experimental Metrics (Beta)" # Commented out problematic tab
        ])
        
        with tab1:
            self.case_type_dashboard()
        
        with tab2:
            self.create_age_demographics()
            
        with tab3:
            self.create_additional_demographics()
            
        with tab4:
            self.create_worldwide_distribution()
            
        with tab5:
            self.create_case_status_insights()
            
        with tab6:
            self.create_corporate_overview()
            
        # Commented out problematic function with datetime handling issues
        # with tab7:
        #    self.create_experimental_metrics()
        
        # The issue with experimental metrics showing up is likely because there's a 
        # partial function definition using triple quotes (""") instead of proper comments (#).
        # To properly hide the experimental metrics, delete the function or comment with # characters.

    def create_age_demographics(self):
        st.title("Age Demographics Analysis")
        
        # Calculate ages and age groups
        if 'Client Date of Birth' in self.airtable_data.columns:
            try:
                # Convert to datetime temporarily for calculation if needed
                dob_series = self.airtable_data['Client Date of Birth']
                
                # Check if we need to parse the date string
                if pd.api.types.is_string_dtype(dob_series.dtype):
                    # Only attempt to calculate for valid date strings
                    valid_date_mask = dob_series.str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)
                    
                    # Create Age column with safe defaults
                    if 'Age' not in self.airtable_data.columns:
                        self.airtable_data['Age'] = np.nan
                    
                    # Define all possible age groups to avoid category modification errors
                    age_groups = ['Unknown', 'Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                    
                    # Create Age Group with all Unknown values first
                    unknown_list = ['Unknown'] * len(self.airtable_data)
                    self.airtable_data['Age Group'] = pd.Categorical(unknown_list, categories=age_groups)
                    
                    # Only calculate ages for valid dates
                    if valid_date_mask.any():
                        self.airtable_data.loc[valid_date_mask, 'Age'] = dob_series[valid_date_mask].apply(self.calculate_age)
                        # Apply get_age_group and ensure result is within allowed categories
                        age_groups_series = self.airtable_data.loc[valid_date_mask, 'Age'].apply(self.get_age_group)
                        self.airtable_data.loc[valid_date_mask, 'Age Group'] = pd.Categorical(age_groups_series, categories=age_groups)
                else:
                    # Original method
                    if 'Age' not in self.airtable_data.columns:
                        self.airtable_data['Age'] = self.airtable_data['Client Date of Birth'].apply(self.calculate_age)
                    
                    # Define all possible age groups to avoid category modification errors
                    age_groups = ['Unknown', 'Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                    
                    # Apply get_age_group to get list-like data
                    age_groups_list = self.airtable_data['Age'].apply(self.get_age_group).tolist()
                    
                    # Create categorical from list
                    self.airtable_data['Age Group'] = pd.Categorical(age_groups_list, categories=age_groups)
            
                # Ensure Age is numeric and handle NaN values
                self.airtable_data['Age'] = pd.to_numeric(self.airtable_data['Age'], errors='coerce').fillna(0)
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age Distribution
                    st.subheader("Age Distribution")
                    
                    # Get valid age groups (exclude Unknown or empty)
                    valid_groups = self.airtable_data['Age Group'][self.airtable_data['Age Group'] != 'Unknown'].value_counts().sort_index()
                    
                    if not valid_groups.empty:
                        fig_age = px.bar(
                            x=valid_groups.index,
                            y=valid_groups.values,
                            title="Age Distribution",
                            labels={'x': 'Age Group', 'y': 'Number of Clients'},
                            color=valid_groups.values,
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_age, use_container_width=True, key="age_distribution")
                        
                        # Summary statistics only for valid ages
                        valid_ages = self.airtable_data['Age'][self.airtable_data['Age'] > 0]
                        if not valid_ages.empty:
                            st.write("Summary Statistics:")
                            st.write(f"Average Age: {valid_ages.mean():.1f} years")
                            st.write(f"Median Age: {valid_ages.median():.1f} years")
                            st.write(f"Age Range: {valid_ages.min():.0f} - {valid_ages.max():.0f} years")
                    else:
                        st.warning("No valid age groups found in the data.")
                
                with col2:
                    # Age Distribution by Main/Dependent Status
                    if 'Main/Dependent' in self.airtable_data.columns:
                        st.subheader("Age Groups by Main/Dependent Status")
                        
                        try:
                            # Convert to standard strings for safer handling - avoids categorical issues
                            age_group_data = self.airtable_data['Age Group'].astype(str)
                            main_dep_data = self.airtable_data['Main/Dependent'].astype(str)
                            
                            # Filter out 'Unknown' age groups
                            valid_mask = (age_group_data != 'Unknown')
                            
                            if valid_mask.any():
                                # Use crosstab with string data
                                age_status_data = pd.crosstab(
                                    age_group_data[valid_mask],
                                    main_dep_data[valid_mask]
                                )
                                
                                # Create plot if we have data
                                if not age_status_data.empty:
                                    fig_age_status = px.bar(
                                        age_status_data,
                                        barmode='group',
                                        title="Age Distribution by Main/Dependent Status",
                                        labels={'value': 'Number of Clients', 'Main/Dependent': 'Status'}
                                    )
                                    st.plotly_chart(fig_age_status, use_container_width=True, key="age_by_status")
                                else:
                                    st.warning("Not enough data for cross-tabulation.")
                            else:
                                st.warning("No valid age data available for this visualization.")
                        except Exception as e:
                            st.error(f"Error creating age by status chart: {str(e)}")
                            st.info("Try uploading a file with more complete age and status data.")
            
            except Exception as e:
                st.error(f"Error in age demographics calculation: {str(e)}")
                st.info("Age demographics could not be calculated from the available data.")
            
        else:
            st.warning("Date of Birth information is not available in the dataset.")

    def create_worldwide_distribution(self):
        """Create a dedicated section for worldwide client citizenship distribution."""
        st.title("Worldwide Client Citizenship Distribution")
        
        # Check if we have the required column
        if 'Country of Citizenship' not in self.airtable_data.columns:
            st.warning("Country of Citizenship data is not available in the dataset.")
            return
            
        try:
            # Clean and validate country data - convert to string first to avoid categorical errors
            if pd.api.types.is_categorical_dtype(self.airtable_data['Country of Citizenship'].dtype):
                # Work with a copy to avoid modifying the original categorical data
                country_data = self.airtable_data['Country of Citizenship'].astype(str)
            else:
                country_data = self.airtable_data['Country of Citizenship'].copy()
                
            # Remove invalid values
            invalid_values = ['All', 'N/A', '', 'nan', 'NaN', 'None', 'Unknown', None]
            country_data = country_data[~country_data.isin(invalid_values)]
            country_data = country_data.dropna()

            if len(country_data) == 0:
                st.warning("No valid country data available for visualization.")
                return

            # Count valid countries
            country_counts = country_data.value_counts()
            
            # Create two columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create the choropleth map
                fig_map = px.choropleth(
                    locations=country_counts.index,
                    locationmode='country names',
                    color=country_counts.values,
                    hover_name=country_counts.index,
                    color_continuous_scale='Viridis',
                    title='Geographic Distribution by Country of Citizenship',
                    labels={'color': 'Number of Cases'}
                )
                
                fig_map.update_layout(
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='equirectangular',
                        showland=True,
                        showcountries=True,
                        landcolor='rgb(243, 243, 243)',
                        countrycolor='rgb(204, 204, 204)'
                    ),
                    width=800,
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_map, use_container_width=True, key="world_map_tab4")

            with col2:
                # Add summary statistics
                st.subheader("Distribution Summary")
                total_clients = len(country_data)
                unique_countries = len(country_counts)
                
                st.metric("Total Clients with Country Data", f"{total_clients:,}")
                st.metric("Number of Unique Countries", f"{unique_countries:,}")
                
                # Show top countries table
                st.subheader("Top 10 Countries")
                top_10_df = pd.DataFrame({
                    'Country': country_counts.head(10).index,
                    'Number of Clients': country_counts.head(10).values
                })
                st.dataframe(
                    top_10_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add percentage of total
                total_cases = len(self.airtable_data)
                coverage = (total_clients / total_cases) * 100
                st.metric(
                    "Data Coverage",
                    f"{coverage:.1f}%",
                    help="Percentage of cases with valid country data"
                )

        except Exception as e:
            st.error(f"Error creating worldwide distribution visualization: {str(e)}")

    def create_world_map(self):
        """Create a world map visualization if valid country data is available."""
        try:
            if 'Country of Birth' not in self.airtable_data.columns:
                return None

            # Clean and validate country data - convert to string first to avoid categorical errors
            if pd.api.types.is_categorical_dtype(self.airtable_data['Country of Birth'].dtype):
                # Work with a copy to avoid modifying the original categorical data
                country_data = self.airtable_data['Country of Birth'].astype(str)
            else:
                country_data = self.airtable_data['Country of Birth'].copy()
                
            # Remove invalid values
            invalid_values = ['All', 'N/A', '', 'nan', 'NaN', 'None', 'Unknown', None]
            country_data = country_data[~country_data.isin(invalid_values)]
            country_data = country_data.dropna()

            if len(country_data) == 0:
                st.warning("No valid country data available for visualization.")
                return None

            # Count valid countries
            country_counts = country_data.value_counts()

            if len(country_counts) == 0:
                return None

            # Create the choropleth map
            fig = px.choropleth(
                locations=country_counts.index,
                locationmode='country names',
                color=country_counts.values,
                hover_name=country_counts.index,
                color_continuous_scale='Viridis',
                title='Geographic Distribution by Country of Birth',
                labels={'color': 'Number of Cases'}
            )
            
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
                width=800,
                height=500
            )

            return fig
        except Exception as e:
            st.error(f"Error creating world map: {str(e)}")
            return None

    def create_additional_demographics(self):
        st.title("Additional Demographics Analysis")
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Distribution (if available)
            if 'Gender' in self.airtable_data.columns:
                st.subheader("Gender Distribution")
                gender_counts = self.airtable_data['Gender'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig_gender, use_container_width=True, key="gender_dist_tab3")
            
            # Main vs Dependent Distribution
            if 'Main/Dependent' in self.airtable_data.columns:
                st.subheader("Main vs Dependent Distribution")
                status_counts = self.airtable_data['Main/Dependent'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Main vs Dependent Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig_status, use_container_width=True, key="status_dist_tab3")

            # World Map (if valid data is available)
            world_map = self.create_world_map()
            if world_map is not None:
                st.subheader("Geographic Distribution")
                st.plotly_chart(world_map, use_container_width=True, key="world_map_tab3")
        
        with col2:
            # Timeline of Cases
            if 'Created Time' in self.airtable_data.columns:
                try:
                    st.subheader("Cases Over Time")
                    # Convert to datetime if not already
                    timeline_data = pd.to_datetime(self.airtable_data['Created Time'], errors='coerce')
                    if not timeline_data.isna().all():
                        timeline_counts = timeline_data.dt.to_period('M').value_counts().sort_index()
                        timeline_counts.index = timeline_counts.index.astype(str)
                        fig_timeline = px.line(
                            x=timeline_counts.index,
                            y=timeline_counts.values,
                            title="Cases Created Over Time",
                            labels={'x': 'Month', 'y': 'Number of Cases'}
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_tab3")
                    else:
                        st.warning("No valid timeline data available.")
                except Exception as e:
                    st.error(f"Error creating timeline: {str(e)}")
            
            # Case Type Distribution
            if 'Case Type' in self.airtable_data.columns:
                try:
                    st.subheader("Case Type Distribution (Top 20)")
                    case_counts = self.airtable_data['Case Type'].value_counts()
                    
                    # Separate top 20 and combine others
                    top_20_cases = case_counts.head(20)
                    others_count = case_counts[20:].sum() if len(case_counts) > 20 else 0
                    
                    if others_count > 0:
                        case_counts_modified = pd.concat([top_20_cases, pd.Series({'Others': others_count})])
                    else:
                        case_counts_modified = top_20_cases

                    if not case_counts_modified.empty:
                        fig_case = px.bar(
                            x=case_counts_modified.index,
                            y=case_counts_modified.values,
                            title="Case Type Distribution (Top 20)",
                            labels={'x': 'Case Type', 'y': 'Number of Cases'},
                            color=case_counts_modified.values,
                            color_continuous_scale='Viridis'
                        )
                        fig_case.update_layout(
                            xaxis_tickangle=-45,
                            showlegend=False,
                            height=500
                        )
                        st.plotly_chart(fig_case, use_container_width=True, key="case_type_dist_tab3")
                    else:
                        st.warning("No case type data available.")
                except Exception as e:
                    st.error(f"Error creating case type distribution: {str(e)}")
                    
    def case_type_dashboard(self):
        st.title("Case Analysis Dashboard")
        
        # Create filtering options in sidebar
        st.sidebar.header("Dashboard Filters")
        
        # Add PERM-specific filtering options
        include_perm = st.sidebar.checkbox("Include PERM Cases", value=True)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Case Type Distribution
            if 'Case Type' in self.airtable_data.columns:
                try:
                    st.subheader("Case Type Distribution (Main Applicants Only)")
                    
                    # Filter for Main applicants only
                    if 'Main/Dependent' in self.airtable_data.columns:
                        main_filter = self.airtable_data['Main/Dependent'].astype(str) == 'Main'
                        main_df = self.airtable_data[main_filter]
                        
                        # Debug PERM case count
                        if 'Case Type' in main_df.columns:
                            perm_count = main_df[main_df['Case Type'].astype(str).str.contains('PERM', case=False, na=False)].shape[0]
                            st.sidebar.write(f"PERM Cases (Main Only): {perm_count}")
                            
                            # Show detailed breakdown
                            st.sidebar.write("PERM Case Types Breakdown:")
                            perm_types = main_df[main_df['Case Type'].astype(str).str.contains('PERM', case=False, na=False)]['Case Type'].value_counts()
                            st.sidebar.write(perm_types)
                            
                            # Option to exclude PERM cases
                            if not include_perm:
                                main_df = main_df[~main_df['Case Type'].astype(str).str.contains('PERM', case=False, na=False)]
                                st.info("PERM cases excluded from visualizations based on filter setting")
                    else:
                        main_df = self.airtable_data
                        st.info("'Main/Dependent' column not found. Showing all cases.")
                    
                    case_counts = main_df['Case Type'].value_counts()
                    
                    # Separate top 20 and combine others
                    top_20_cases = case_counts.head(20)
                    others_count = case_counts[20:].sum() if len(case_counts) > 20 else 0
                    
                    if others_count > 0:
                        case_counts_modified = pd.concat([top_20_cases, pd.Series({'Others': others_count})])
                    else:
                        case_counts_modified = top_20_cases

                    if not case_counts_modified.empty:
                        fig_case = px.bar(
                            x=case_counts_modified.index,
                            y=case_counts_modified.values,
                            title="Case Type Distribution (Top 20, Main Applicants Only)",
                            labels={'x': 'Case Type', 'y': 'Number of Main Cases'},
                            color=case_counts_modified.values,
                            color_continuous_scale='Viridis'
                        )
                        fig_case.update_layout(
                            xaxis_tickangle=-45,
                            showlegend=False,
                            height=500
                        )
                        st.plotly_chart(fig_case, use_container_width=True, key="case_type_dist_tab1")
                    else:
                        st.warning("No case type data available.")
                except Exception as e:
                    st.error(f"Error creating case type distribution: {str(e)}")
            
            # Add export options
            if st.button("Export to CSV"):
                try:
                    csv = self.airtable_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="case_data.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting to CSV: {str(e)}")
        
        with col2:
            # Cases by Year Overview
            st.subheader("Cases by Year (Main Applicants Only)")
            try:
                if 'Created Time' in self.airtable_data.columns:
                    # Filter for Main applicants only
                    if 'Main/Dependent' in self.airtable_data.columns:
                        main_filter = self.airtable_data['Main/Dependent'].astype(str) == 'Main'
                        main_df = self.airtable_data[main_filter]
                    else:
                        main_df = self.airtable_data
                        st.info("'Main/Dependent' column not found. Showing all cases.")
                    
                    # Convert to datetime if not already
                    created_dates = pd.to_datetime(main_df['Created Time'], errors='coerce')
                    
                    if not created_dates.isna().all():
                        # Extract year and count cases per year
                        yearly_counts = created_dates.dt.year.value_counts().sort_index()
                        
                        # Create DataFrame for visualization
                        yearly_df = pd.DataFrame({
                            'Year': yearly_counts.index,
                            'Number of Main Cases': yearly_counts.values
                        })
                        
                        # Create bar chart
                        fig_yearly = px.bar(
                            yearly_df,
                            x='Year',
                            y='Number of Main Cases',
                            title="Number of Main Cases by Year",
                            labels={'Year': 'Year', 'Number of Main Cases': 'Number of Main Cases'},
                            color='Number of Main Cases',
                            color_continuous_scale='Viridis'
                        )
                        
                        # Update layout
                        fig_yearly.update_layout(
                            xaxis=dict(
                                tickmode='linear',
                                dtick=1,  # Show every year
                                tickangle=0
                            ),
                            showlegend=False,
                            height=400
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig_yearly, use_container_width=True, key="yearly_cases_tab1")
                        
                        # Add summary statistics
                        total_main_cases = len(main_df)
                        st.write(f"Total Main Cases: {total_main_cases:,}")
                        if len(yearly_counts) > 0:
                            st.write(f"Year Range: {yearly_counts.index.min()} - {yearly_counts.index.max()}")
                            avg_cases_per_year = total_main_cases / len(yearly_counts)
                            st.write(f"Average Main Cases per Year: {avg_cases_per_year:,.0f}")
                    else:
                        st.warning("No valid creation dates available.")
                        # Show only total cases if dates are not available
                        if 'Main/Dependent' in self.airtable_data.columns:
                            main_filter = self.airtable_data['Main/Dependent'].astype(str) == 'Main'
                            total_main_cases = main_filter.sum()
                            st.metric("Total Main Cases", f"{total_main_cases:,}")
                        else:
                            total_cases = len(self.airtable_data)
                            st.metric("Total Cases", f"{total_cases:,}")
                else:
                    st.warning("Creation date information is not available.")
                    # Show only total cases if dates are not available
                    if 'Main/Dependent' in self.airtable_data.columns:
                        main_filter = self.airtable_data['Main/Dependent'].astype(str) == 'Main'
                        total_main_cases = main_filter.sum()
                        st.metric("Total Main Cases", f"{total_main_cases:,}")
                    else:
                        total_cases = len(self.airtable_data)
                        st.metric("Total Cases", f"{total_cases:,}")
                
                # Always show unique case types count for main applicants
                if 'Case Type' in self.airtable_data.columns and 'Main/Dependent' in self.airtable_data.columns:
                    main_filter = self.airtable_data['Main/Dependent'].astype(str) == 'Main'
                    main_df = self.airtable_data[main_filter]
                    st.metric("Unique Case Types (Main Applicants)", f"{main_df['Case Type'].nunique():,}")
                elif 'Case Type' in self.airtable_data.columns:
                    st.metric("Unique Case Types", f"{self.airtable_data['Case Type'].nunique():,}")
                
            except Exception as e:
                st.error(f"Error creating cases by year overview: {str(e)}")
                # Show basic statistics in case of error
                if 'Main/Dependent' in self.airtable_data.columns:
                    main_filter = self.airtable_data['Main/Dependent'].astype(str) == 'Main'
                    total_main_cases = main_filter.sum()
                    st.metric("Total Main Cases", f"{total_main_cases:,}")
                else:
                    total_cases = len(self.airtable_data)
                    st.metric("Total Cases", f"{total_cases:,}")
            
            # Export to PDF option
            if st.button("Export to PDF"):
                try:
                    pdf_path = self.generate_pdf_report()
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF Report",
                            data=f,
                            file_name="case_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(pdf_path)  # Clean up temporary file
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")

    def create_case_status_insights(self):
        """Create visualizations for case status and progress insights."""
        st.title("Case Status and Progress Insights")
        
        try:
            # Create a clean copy of the dataframe
            df = self.airtable_data.copy()
            
            # Filter for Main applicants only
            if 'Main/Dependent' in df.columns:
                main_filter = df['Main/Dependent'].astype(str) == 'Main'
                main_df = df[main_filter]
                df = main_df  # Use only Main applicants for all visualizations
                st.info("Showing data for Main applicants only")
            else:
                st.info("'Main/Dependent' column not found. Showing all cases.")
            
            # Create three columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Case Status Distribution
                if 'Case Status' in df.columns:
                    st.subheader("Case Status Distribution (Main Applicants)")
                    status_counts = df['Case Status'].value_counts()
                    fig_status = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Distribution by Case Status (Main Applicants)",
                        hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_status, use_container_width=True, key="case_status_dist")

                # Processing Type Distribution
                if 'Processing Type' in df.columns:
                    st.subheader("Processing Type Distribution (Main Applicants)")
                    proc_counts = df['Processing Type'].value_counts()
                    fig_proc = px.bar(
                        x=proc_counts.index,
                        y=proc_counts.values,
                        title="Distribution by Processing Type (Main Applicants)",
                        color=proc_counts.values,
                        color_continuous_scale='Viridis',
                        labels={'x': 'Processing Type', 'y': 'Number of Main Cases'}
                    )
                    fig_proc.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_proc, use_container_width=True, key="processing_type_dist")

            with col2:
                # Active vs Inactive Cases
                if 'Is Case Active?' in df.columns:
                    st.subheader("Active vs Inactive Cases (Main Applicants)")
                    active_counts = df['Is Case Active?'].value_counts()
                    fig_active = px.pie(
                        values=active_counts.values,
                        names=active_counts.index,
                        title="Active vs Inactive Cases (Main Applicants)",
                        color_discrete_sequence=['#2ecc71', '#e74c3c']
                    )
                    st.plotly_chart(fig_active, use_container_width=True, key="active_cases_dist")

                # Petition Category Analysis
                if 'Petition Category' in df.columns:
                    st.subheader("Petition Category Analysis (Main Applicants)")
                    pet_counts = df['Petition Category'].value_counts().head(10)
                    fig_pet = px.bar(
                        x=pet_counts.values,
                        y=pet_counts.index,
                        orientation='h',
                        title="Top 10 Petition Categories (Main Applicants)",
                        color=pet_counts.values,
                        color_continuous_scale='Viridis',
                        labels={'x': 'Number of Main Cases', 'y': 'Petition Category'}
                    )
                    st.plotly_chart(fig_pet, use_container_width=True, key="petition_cat_dist")

            # Timeline of Case Statuses
            if 'Case Open Date' in df.columns and 'Case Status' in df.columns:
                st.subheader("Case Status Timeline (Main Applicants)")
                try:
                    # Convert dates to datetime for grouping
                    df['Case Open Date'] = pd.to_datetime(df['Case Open Date'], errors='coerce')
                    
                    # Create timeline data using pivot table instead of groupby
                    timeline_data = pd.pivot_table(
                        df,
                        index=pd.Grouper(key='Case Open Date', freq='ME'),
                        columns='Case Status',
                        aggfunc='size',
                        fill_value=0
                    ).reset_index()
                    
                    # Melt the data for plotting
                    timeline_data = timeline_data.melt(
                        id_vars=['Case Open Date'],
                        var_name='Case Status',
                        value_name='count'
                    )
                    
                    fig_timeline = px.line(
                        timeline_data,
                        x='Case Open Date',
                        y='count',
                        color='Case Status',
                        title="Case Status Timeline (Main Applicants)",
                        labels={'count': 'Number of Main Cases', 'Case Open Date': 'Date'}
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True, key="case_status_timeline")
                except Exception as e:
                    st.warning(f"Could not create timeline visualization: {str(e)}")

        except Exception as e:
            st.error(f"Error creating case status insights: {str(e)}")

    def create_corporate_overview(self):
        """Create visualizations for corporate-level insights."""
        st.title("Corporate Overview")
        
        try:
            # Create a clean copy of the dataframe
            df = self.airtable_data.copy()
            
            # Filter for Main applicants only
            if 'Main/Dependent' in df.columns:
                main_filter = df['Main/Dependent'].astype(str) == 'Main'
                main_df = df[main_filter]
                df = main_df  # Use only Main applicants for all visualizations
                st.info("Showing data for Main applicants only")
            else:
                st.info("'Main/Dependent' column not found. Showing all cases.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Corporation Type Distribution
                if 'Corporation Type' in df.columns:
                    st.subheader("Corporation Type Distribution (Main Applicants)")
                    corp_type_counts = df['Corporation Type'].value_counts()
                    fig_corp = px.pie(
                        values=corp_type_counts.values,
                        names=corp_type_counts.index,
                        title="Distribution by Corporation Type (Main Applicants)",
                        hole=0.3
                    )
                    st.plotly_chart(fig_corp, use_container_width=True, key="corp_type_dist")

                # Department/Business Unit Analysis
                if 'Department/Business Unit' in df.columns:
                    st.subheader("Department Distribution (Main Applicants)")
                    dept_counts = df['Department/Business Unit'].value_counts().head(10)
                    fig_dept = px.bar(
                        x=dept_counts.index,
                        y=dept_counts.values,
                        title="Top 10 Departments/Business Units (Main Applicants)",
                        color=dept_counts.values,
                        color_continuous_scale='Viridis',
                        labels={'x': 'Department/Business Unit', 'y': 'Number of Main Cases'}
                    )
                    fig_dept.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_dept, use_container_width=True, key="dept_dist")

            with col2:
                # Service Center Analysis
                if 'Service Center' in df.columns:
                    st.subheader("Service Center Distribution (Main Applicants)")
                    service_counts = df['Service Center'].value_counts()
                    fig_service = px.pie(
                        values=service_counts.values,
                        names=service_counts.index,
                        title="Distribution by Service Center (Main Applicants)"
                    )
                    st.plotly_chart(fig_service, use_container_width=True, key="service_center_dist")

                # Corporation Status
                if 'Corporation Status' in df.columns:
                    st.subheader("Corporation Status Overview (Main Applicants)")
                    corp_status_counts = df['Corporation Status'].value_counts()
                    fig_status = px.bar(
                        x=corp_status_counts.index,
                        y=corp_status_counts.values,
                        title="Corporation Status Distribution (Main Applicants)",
                        color=corp_status_counts.values,
                        color_continuous_scale='Viridis',
                        labels={'x': 'Corporation Status', 'y': 'Number of Main Cases'}
                    )
                    st.plotly_chart(fig_status, use_container_width=True, key="corp_status_dist")

            # Cross Analysis using pivot table instead of groupby
            if all(col in df.columns for col in ['Corporation Type', 'Case Status']):
                st.subheader("Case Status by Corporation Type (Main Applicants)")
                # Create pivot table
                cross_tab = pd.pivot_table(
                    df,
                    index='Corporation Type',
                    columns='Case Status',
                    aggfunc='size',
                    fill_value=0
                )
                
                fig_cross = px.bar(
                    cross_tab,
                    title="Case Status Distribution by Corporation Type (Main Applicants)",
                    barmode='group',
                    labels={'value': 'Number of Main Cases'}
                )
                fig_cross.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_cross, use_container_width=True, key="corp_case_status_dist")

        except Exception as e:
            st.error(f"Error creating corporate overview: {str(e)}")

if __name__ == "__main__":
    dashboard = TestDashboard()
    dashboard.run_dashboard() 