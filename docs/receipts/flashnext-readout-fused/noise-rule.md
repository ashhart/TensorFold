| invocation | prompt | streams | M1 | M2 | U1 | U2 | F1 | F2 | U vs main | F vs main | tokens a round M / U / F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ttft | 2k | 1 | 1.05 | 1.08 | 1.06 | 1.07 | 1.05 | 1.03 | +0.5% level | -1.8% level | - |
| ttft | 8k | 1 | 4.01 | 3.99 | 3.98 | 4.00 | 3.87 | 3.85 | -0.3% level | -3.5% faster | - |
| ttft | 32k | 1 | 15.90 | 16.06 | 16.03 | 16.02 | 15.51 | 15.54 | +0.3% level | -2.8% faster | - |
| ttft | 64k | 1 | 32.80 | 33.11 | 33.12 | 32.96 | 31.87 | 32.02 | +0.3% level | -3.1% faster | - |
| cache | 2571 tokens | 1 | 1.233 | 1.223 | 1.219 | 1.219 | 1.183 | 1.185 | -0.7% level | -3.6% faster | - |
| openai_t0 | fibonacci-raw | 1 | 72.6 | 72.0 | 72.2 | 72.1 | 72.1 | 72.1 | -0.2% level | -0.2% level | 5.224 / 5.224 / 5.224 |
| openai_t0 | gpu-chat-no-think | 1 | 46.6 | 46.4 | 46.5 | 46.4 | 46.4 | 46.4 | -0.1% level | -0.2% level | 2.813 / 2.813 / 2.813 |
| conc_t0 | chat | 1 | 46.7 | 46.5 | 46.5 | 46.4 | 46.5 | 46.4 | -0.3% level | -0.3% level | 2.813 / 2.813 / 2.813 |
| conc_t0 | chat | 2 | 85.7 | 85.5 | 85.6 | 85.6 | 85.7 | 85.6 | +0.0% level | +0.1% level | 2.510 / 2.510 / 2.510 |
| conc_t0 | chat | 4 | 152.6 | 152.3 | 152.6 | 152.1 | 152.3 | 152.2 | -0.1% level | -0.1% level | 2.510 / 2.510 / 2.510 |
| conc_t0 | code | 1 | 72.4 | 72.3 | 72.1 | 72.2 | 72.5 | 72.3 | -0.3% level | +0.1% level | 5.224 / 5.224 / 5.224 |
| conc_t0 | code | 2 | 138.0 | 137.4 | 137.4 | 137.4 | 137.5 | 137.7 | -0.2% level | -0.1% level | 4.830 / 4.830 / 4.830 |
| conc_t0 | code | 4 | 242.4 | 241.3 | 242.1 | 241.5 | 241.2 | 241.4 | -0.0% level | -0.2% level | 4.830 / 4.830 / 4.830 |
| conc_t1_s1234 | chat | 1 | 44.6 | 44.4 | 44.5 | 44.4 | 44.4 | 44.5 | -0.1% level | -0.1% level | 2.393 / 2.393 / 2.393 |
| conc_t1_s1234 | chat | 2 | 69.5 | 69.0 | 69.3 | 69.0 | 69.3 | 69.4 | -0.1% level | +0.1% level | 2.450 / 2.450 / 2.450 |
| conc_t1_s1234 | chat | 4 | 102.9 | 102.7 | 103.0 | 102.7 | 102.9 | 102.6 | +0.0% level | -0.0% level | 2.456 / 2.456 / 2.456 |
| conc_t1_s1234 | code | 1 | 70.3 | 69.9 | 70.0 | 69.8 | 69.9 | 70.2 | -0.3% level | -0.1% level | 4.571 / 4.571 / 4.571 |
| conc_t1_s1234 | code | 2 | 88.5 | 88.3 | 88.3 | 88.1 | 88.4 | 88.2 | -0.2% level | -0.1% level | 3.765 / 3.765 / 3.765 |
| conc_t1_s1234 | code | 4 | 129.2 | 128.5 | 128.8 | 128.7 | 128.9 | 128.7 | -0.1% level | -0.0% level | 3.835 / 3.835 / 3.835 |

Server acceptance (mtp_accepted_total over mtp_drafted_total, per invocation):

- openai_t0: M1 1488/2384 = 0.6242, U1 1488/2384 = 0.6242, F1 1488/2384 = 0.6242, M2 1488/2384 = 0.6242, F2 1488/2384 = 0.6242, U2 1488/2384 = 0.6242
- conc_t0: M1 7934/11682 = 0.6792, U1 7934/11682 = 0.6792, F1 7934/11682 = 0.6792, M2 7934/11682 = 0.6792, F2 7934/11682 = 0.6792, U2 7934/11682 = 0.6792
- conc_t1_s1234: M1 8607/11917 = 0.7222, U1 8607/11917 = 0.7222, F1 8607/11917 = 0.7222, M2 8607/11917 = 0.7222, F2 8607/11917 = 0.7222, U2 8607/11917 = 0.7222

Memory after each arm (docker stats) and the pool line:

- M1: 4.33GiB / 123.7GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve
- U1: 4.324GiB / 123.7GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve
- F1: 4.346GiB / 123.7GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve
- M2: 4.32GiB / 123.7GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve
- F2: 4.336GiB / 123.7GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve
- U2: 4.327GiB / 123.7GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 28.0 GiB left after a 9.0 GiB reserve
