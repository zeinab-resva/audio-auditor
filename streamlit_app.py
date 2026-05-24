if energy > energy_threshold and spectral_val > centroid_threshold and rolloff_val > rolloff_threshold:
                        severity = "High 🚨" if energy > (energy_threshold * 1.8) else "Medium ⚠️"
                        violations.append({
                            "Timestamp ⏱️": timestamp,
                            "Audio Status": "Confirmed Background Noise / Distortion",
                            "Severity Level 📊": severity
                        })
                
                st.markdown("### 💡 Audit Summary & Results")
                if len(violations) > 0:
                    df = pd.DataFrame(violations)
                    df = df.drop_duplicates(subset=['Timestamp ⏱️'])
                    st.error(f"🚨 Detected {len(df)} seconds of background noise (Agent's voice skipped successfully).")
                    st.markdown("### 📊 Violation Log Table")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("✅ Audit Passed: Call complies with quality standards. Agent voice is clear, no background noise detected.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

st.markdown("---")
st.caption("Smart Audit Tool - Free and Unlimited Use.")
