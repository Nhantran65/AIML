test_X = test_df.values / 255.
rfc_pred = model.predict(test_X)
gnb_pred = gnb.predict(test_X)


sub = pd.read_csv('../input/sample_submission.csv')
sub.head()

# Make submission file
sub['Label'] = rfc_pred
sub.to_csv('submission.csv', index=False)

# Make NB submission file
sub['Label'] = gnb_pred
sub.to_csv('GNB_submission.csv', index=False)

# Show our submission file
sub.head(10)