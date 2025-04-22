import pandas as pd
import streamlit as st
import numpy as np
import pyairtable
import dotenv
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile
import pycountry
dotenv.load_dotenv()

airtable_api_key = os.getenv("AIRTABLE_API_KEY")
airtable_base_id = os.getenv("AIRTABLE_BASE_ID")
airtable_table_name = os.getenv("AIRTABLE_TABLE_NAME")


class GrabAirTableData:
    def __init__(self, airtable_api_key, airtable_base_id, airtable_table_name):
        self.airtable_api_key = airtable_api_key
        self.airtable_base_id = airtable_base_id
        self.airtable_table_name = airtable_table_name

    def get_airtable_data(self):
        airtable = pyairtable.Api(self.airtable_api_key)
        table = airtable.table(self.airtable_base_id, self.airtable_table_name)
        records = table.all()
        df = pd.DataFrame(records)
        return df
    
class CreateDashboard(GrabAirTableData):
    def __init__(self, airtable_data):
        self.airtable_data = airtable_data
        self.age_ranges = {
            "0-10": (0, 10),
            "11-20": (11, 20),
            "21-30": (21, 30),
            "31-40": (31, 40),
            "41-50": (41, 50),
            "51-60": (51, 60),
            "61-70": (61, 70),
            "71+": (71, float('inf'))
        }
        # Convert date columns to datetime if they exist
        if 'Client Date of Birth' in self.airtable_data.columns:
            self.airtable_data['Client Date of Birth'] = pd.to_datetime(self.airtable_data['Client Date of Birth'])
        if 'Created Time' in self.airtable_data.columns:
            self.airtable_data['Created Time'] = pd.to_datetime(self.airtable_data['Created Time'])
        
        # Initialize session state for filters
        if 'filters' not in st.session_state:
            st.session_state.filters = {
                'date_range': None,
                'case_type': 'All',
                'country': 'All',
                'status': 'All'
            }

    def calculate_age(self, dob):
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age

    def get_age_group(self, age):
        for group, (min_age, max_age) in self.age_ranges.items():
            if min_age <= age <= max_age:
                return group
        return "Unknown"

    def get_country_coordinates(self, country_name):
        try:
            # Try to get the country code
            country = pycountry.countries.get(name=country_name)
            if not country:
                # Try searching by common name
                country = pycountry.countries.search(country_name)[0]
            
            # Get the country's alpha-2 code
            country_code = country.alpha_2
            
            # Use a simple mapping of country codes to approximate center points
            # This is a simplified version - in production, you might want to use a more accurate geocoding service
            country_centers = {
                'US': (37.0902, -95.7129),  # United States
                'CA': (56.1304, -106.3468),  # Canada
                'GB': (55.3781, -3.4360),    # United Kingdom
                'AU': (-25.2744, 133.7751),  # Australia
                'IN': (20.5937, 78.9629),    # India
                'CN': (35.8617, 104.1954),   # China
                'BR': (-14.2350, -51.9253),  # Brazil
                'DE': (51.1657, 10.4515),    # Germany
                'FR': (46.2276, 2.2137),     # France
                'IT': (41.8719, 12.5674),    # Italy
                'ES': (40.4637, -3.7492),    # Spain
                'JP': (36.2048, 138.2529),   # Japan
                'KR': (35.9078, 127.7669),   # South Korea
                'MX': (23.6345, -102.5528),  # Mexico
                'RU': (61.5240, 105.3188),   # Russia
            }
            
            return country_centers.get(country_code, (0, 0))
        except:
            return (0, 0)  # Default to (0,0) if country not found

    def create_world_map(self):
        st.title("Global Client Distribution")
        
        # Count clients by country
        country_counts = self.airtable_data['Country of Citizenship'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Client Count']
        
        # Add coordinates for each country
        country_counts['coordinates'] = country_counts['Country'].apply(self.get_country_coordinates)
        country_counts[['lat', 'lon']] = pd.DataFrame(country_counts['coordinates'].tolist(), index=country_counts.index)
        
        # Create the map
        fig = px.scatter_geo(
            country_counts,
            lat='lat',
            lon='lon',
            size='Client Count',
            color='Client Count',
            hover_name='Country',
            hover_data=['Client Count'],
            projection='natural earth',
            title='Client Distribution by Country',
            color_continuous_scale='Viridis',
            size_max=50
        )
        
        # Update layout
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        # Display the map with a unique key
        st.plotly_chart(fig, use_container_width=True, key="main_world_map")
        
        # Add a summary table below the map
        st.subheader("Client Count by Country")
        st.dataframe(
            country_counts[['Country', 'Client Count']].sort_values('Client Count', ascending=False),
            use_container_width=True
        )

    def create_filters(self):
        st.sidebar.title("Filters")
        
        # Date range filter
        if 'Created Time' in self.airtable_data.columns:
            min_date = self.airtable_data['Created Time'].min()
            max_date = self.airtable_data['Created Time'].max()
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            st.session_state.filters['date_range'] = date_range
            if len(date_range) == 2:
                self.airtable_data = self.airtable_data[
                    (self.airtable_data['Created Time'].dt.date >= date_range[0]) &
                    (self.airtable_data['Created Time'].dt.date <= date_range[1])
                ]
        
        # Case Type filter
        if 'Case Type' in self.airtable_data.columns:
            case_types = ['All'] + list(self.airtable_data['Case Type'].unique())
            selected_case_type = st.sidebar.selectbox(
                "Select Case Type",
                case_types,
                index=case_types.index(st.session_state.filters['case_type'])
            )
            st.session_state.filters['case_type'] = selected_case_type
            if selected_case_type != 'All':
                self.airtable_data = self.airtable_data[self.airtable_data['Case Type'] == selected_case_type]
        
        # Country filter
        if 'Country of Citizenship' in self.airtable_data.columns:
            countries = ['All'] + list(self.airtable_data['Country of Citizenship'].unique())
            selected_country = st.sidebar.selectbox(
                "Select Country",
                countries,
                index=countries.index(st.session_state.filters['country'])
            )
            st.session_state.filters['country'] = selected_country
            if selected_country != 'All':
                self.airtable_data = self.airtable_data[self.airtable_data['Country of Citizenship'] == selected_country]
        
        # Main/Dependent filter
        if 'Main/Dependent' in self.airtable_data.columns:
            status_options = ['All'] + list(self.airtable_data['Main/Dependent'].unique())
            selected_status = st.sidebar.selectbox(
                "Select Main/Dependent Status",
                status_options,
                index=status_options.index(st.session_state.filters['status'])
            )
            st.session_state.filters['status'] = selected_status
            if selected_status != 'All':
                self.airtable_data = self.airtable_data[self.airtable_data['Main/Dependent'] == selected_status]

    def create_interactive_charts(self):
        # Create interactive Plotly charts
        if 'Case Type' in self.airtable_data.columns:
            fig_case = px.pie(
                self.airtable_data,
                names='Case Type',
                title='Case Type Distribution',
                hole=0.3
            )
            fig_case.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            st.plotly_chart(fig_case, use_container_width=True, key="case_type_pie")
            
            # Add click-to-filter functionality
            selected_points = st.session_state.get('selected_points', [])
            if selected_points:
                selected_case = selected_points[0]['x']
                st.session_state.filters['case_type'] = selected_case
                st.experimental_rerun()
        
        if 'Country of Citizenship' in self.airtable_data.columns:
            # Fill empty/null values with 'Others or Not Declared'
            country_data = self.airtable_data.copy()
            country_data['Country of Citizenship'] = country_data['Country of Citizenship'].fillna('Others or Not Declared')
            country_data['Country of Citizenship'] = country_data['Country of Citizenship'].replace('', 'Others or Not Declared')
            
            # Get country counts and sort by count descending
            country_counts = country_data['Country of Citizenship'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            
            # Create bar chart
            fig_country = px.bar(
                country_counts,
                x='Country',
                y='Count',
                title='Country Distribution',
                labels={'Count': 'Number of Cases', 'Country': 'Country of Citizenship'},
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            # Update layout for better readability
            fig_country.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                margin=dict(b=100)  # Add bottom margin for rotated labels
            )
            
            # Add hover template
            fig_country.update_traces(
                hovertemplate='<b>%{x}</b><br>Cases: %{y}<extra></extra>'
            )
            
            st.plotly_chart(fig_country, use_container_width=True, key="country_bar")

    def generate_pdf_report(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Case Analysis Report", ln=True, align='C')
        pdf.ln(10)
        
        # Add filters information
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Applied Filters:", ln=True)
        pdf.set_font("Arial", size=10)
        for key, value in st.session_state.filters.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
        pdf.ln(10)
        
        # Add summary statistics
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Summary Statistics:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Total Cases: {len(self.airtable_data)}", ln=True)
        pdf.cell(200, 10, txt=f"Unique Case Types: {self.airtable_data['Case Type'].nunique()}", ln=True)
        pdf.cell(200, 10, txt=f"Countries Represented: {self.airtable_data['Country of Citizenship'].nunique()}", ln=True)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf.output(tmp.name)
            return tmp.name

    def create_age_dashboard(self):
        st.title("Age Distribution Analysis")
        
        # Apply filters
        self.create_filters()
        
        # Calculate ages and age groups
        self.airtable_data['Age'] = self.airtable_data['Client Date of Birth'].apply(self.calculate_age)
        self.airtable_data['Age Group'] = self.airtable_data['Age'].apply(self.get_age_group)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            age_counts = self.airtable_data['Age Group'].value_counts().sort_index()
            st.bar_chart(age_counts)
            
            # Display summary statistics
            st.write("Summary Statistics:")
            st.write(f"Average Age: {self.airtable_data['Age'].mean():.1f} years")
            st.write(f"Median Age: {self.airtable_data['Age'].median():.1f} years")
            st.write(f"Age Range: {self.airtable_data['Age'].min()} - {self.airtable_data['Age'].max()} years")
        
        with col2:
            st.subheader("Age Distribution by Main/Dependent Status")
            pivot_data = pd.pivot_table(
                self.airtable_data,
                values='Age',
                index='Age Group',
                columns='Main/Dependent',
                aggfunc='count',
                fill_value=0
            )
            st.bar_chart(pivot_data)

    def case_type_dashboard(self):
        st.title("Case Analysis Dashboard")
        
        # Apply filters
        self.create_filters()
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Case Analysis", "Global Distribution"])
        
        with tab1:
            # Create interactive charts
            self.create_interactive_charts()
            
            # Add export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export to CSV"):
                    csv = self.airtable_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="case_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Export to PDF"):
                    pdf_path = self.generate_pdf_report()
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF Report",
                            data=f,
                            file_name="case_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(pdf_path)  # Clean up temporary file
        
        with tab2:
            self.create_world_map()












