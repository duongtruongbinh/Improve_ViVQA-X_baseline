# ... (previous code) ...
            # The detailed summary is now logged inside evaluate_comprehensive -> _log_evaluation_summary
            # No need for separate logging here.
            
        except Exception as eval_error:
            logging.error(f"❌ Evaluation failed: {eval_error}")
            import traceback
            logging.error(f"Evaluation traceback: {traceback.format_exc()}")

    else:
        # If evaluation is disabled, print a simpler summary.
        logging.info(f"📊 Evaluation disabled. Finalizing run...")
        # The new summary table is the primary output, so we can keep this part minimal.
        
    logging.info("🎉 FDR Pipeline completed.")
    
    # Display final evaluation table if evaluation was enabled
    if enable_evaluation and 'evaluation_results' in locals():
        _print_evaluation_table(evaluation_results)

    return results
        
    except FileNotFoundError as e:
# ... (rest of the code) ...
# ... (previous code) ...
if __name__ == "__main__":
    results = main()
    # The final summary is now handled by the pipeline's table view.
    # This provides a clean exit point.
    if results:
        print(f"\n✅ Run finished. Processed {len(results)} items.")
    else:
        print("\n❌ Run finished with errors.") 