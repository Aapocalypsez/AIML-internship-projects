# Data Dictionary — IEEE-CIS Fraud Detection

This documents the key columns in the IEEE-CIS dataset.
The dataset is anonymized — most column names are obfuscated. Meanings below
are derived from Kaggle discussions and the competition description.

## Transaction Table (`train_transaction.csv`)

### Core Fields
| Column | Type | Description |
|---|---|---|
| `TransactionID` | int | Unique transaction identifier |
| `isFraud` | int (0/1) | **Target variable** — 1 if fraudulent |
| `TransactionDT` | int | Seconds since a reference date (2017-12-01 based on Kaggle discussions) |
| `TransactionAmt` | float | Transaction amount in USD |
| `ProductCD` | str | Product code (W, H, C, S, R) — type of product purchased |

### Card Information (card1–card6)
| Column | Type | Description |
|---|---|---|
| `card1` | int | Card number hash — can be used as partial user identifier |
| `card2` | float | Card attribute (possibly expiry-related) |
| `card3` | float | Card attribute |
| `card4` | str | Card brand: visa, mastercard, discover, american express |
| `card5` | float | Card attribute |
| `card6` | str | Card type: debit, credit, debit or credit, charge card |

### Address Information
| Column | Type | Description |
|---|---|---|
| `addr1` | float | Billing address — partial zip code or region code |
| `addr2` | float | Billing country code (87 = US for most rows) |
| `dist1` | float | Distance between billing and shipping addresses |
| `dist2` | float | Another distance metric |

### Email Domains
| Column | Type | Description |
|---|---|---|
| `P_emaildomain` | str | Purchaser email domain (gmail.com, yahoo.com, etc.) |
| `R_emaildomain` | str | Recipient email domain |

### Counting Features (C1–C14)
Counting variables related to the card's usage history. Exact definitions are anonymized.
Higher values generally indicate more activity on the card.

| Column | Likely Meaning |
|---|---|
| `C1`–`C6` | Various counts of prior transactions on this card/address |
| `C7`–`C14` | Additional counting features |

### Time Delta Features (D1–D15)
Days between events. These are key for building the pseudo-user key and detecting velocity.

| Column | Description |
|---|---|
| `D1` | **Days since account opened** — used to derive `account_start_day` |
| `D2`–`D15` | Days between various other events (anonymized) |

### Vesta Features (V1–V339)
Engineered features from Vesta (the competition host). Meanings are not public.
These are grouped into related blocks that correspond to different aspects of
the transaction risk assessment.

| Block | Columns | Likely Domain |
|---|---|---|
| V1–V11 | 11 features | Card verification / matching |
| V12–V34 | 23 features | Transaction similarity metrics |
| V35–V52 | 18 features | Time-based patterns |
| V53–V74 | 22 features | Count-based features |
| V75–V94 | 20 features | Amount-based features |
| V95–V137 | 43 features | Extended risk scoring |
| V138–V166 | 29 features | Device/browser patterns |
| V167–V216 | 50 features | Historical behavior |
| V217–V278 | 62 features | Network features |
| V279–V339 | 61 features | Misc engineered features |

### Match Features (M1–M9)
Boolean/categorical match indicators. Whether certain fields match between the
transaction and the card on file.

| Column | Likely Meaning |
|---|---|
| `M1`–`M3` | Name/address match flags |
| `M4` | Match level: M0, M1, M2 |
| `M5`–`M9` | Additional verification match flags |

## Identity Table (`train_identity.csv`)

| Column | Type | Description |
|---|---|---|
| `TransactionID` | int | Join key to transaction table |
| `id_01`–`id_11` | float | Numerical identity features (anonymized) |
| `id_12`–`id_38` | str | Categorical identity features |
| `DeviceType` | str | mobile, desktop |
| `DeviceInfo` | str | Device make/model string |

## Pseudo-User Key (Our Construction)

| Column | Derivation | Purpose |
|---|---|---|
| `pseudo_user_id` | `card1 + "_" + addr1 + "_" + account_start_day` | Groups transactions by likely cardholder |
| `account_start_day` | `(TransactionDT - D1 * 86400) // 86400` | Estimates when the account was opened |

## Dataset Statistics
- **Total transactions**: ~590,540
- **Fraud rate**: ~3.49%
- **Time span**: ~182 days (Dec 2017 – May 2018)
- **Identity table coverage**: ~26% of transactions have identity data
- **Missingness**: High in V-columns (up to 85%), moderate in D-columns
