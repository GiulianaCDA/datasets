// Validate a single transaction
const transactionId = "txid123";
const isValid = await isValidSlpTxid(transactionId);

// Validate a list of transactions
const transactionIds = ["txid456", "txid789"];
const validTransactionIds = await validateSlpTransactions(transactionIds);
