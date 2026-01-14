import pandas as pd
import os
from src.logger import logging_instance

class DataIngestor:
    def __init__(self, energy_path, temp_path, output_dir):
        """
        Data Ingestion klassi: Ma'lumotlarni yuklaydi, tozalaydi va birlashtiradi.
        """
        self.energy_path = energy_path
        self.temp_path = temp_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_ingestion(self):
        try:
            logging_instance.info("1-Bosqich: Data Ingestion boshlandi.")

            # 1. Ma'lumotlarni yuklash
            energy = pd.read_csv(self.energy_path)
            temp = pd.read_csv(self.temp_path)
            logging_instance.info(f"Fayllar yuklandi. Energy: {energy.shape}, Temp: {temp.shape}")

            # 2. Vaqt formatiga o'tkazish
            energy['Datetime'] = pd.to_datetime(energy['Datetime'])
            temp['datetime'] = pd.to_datetime(temp['datetime'])

            # 3. Energy Data: Dublikatlarni o'chirish va Vaqt uzilishlarini to'ldirish
            energy = energy.set_index('Datetime').sort_index()
            
            # Dublikat indexlarni tekshirish (Masalan: Daylight Saving Time tufayli)
            if energy.index.duplicated().any():
                dup_count = energy.index.duplicated().sum()
                logging_instance.warning(f"Energy ma'lumotida {dup_count} ta dublikat vaqt topildi va o'chirildi.")
                energy = energy[~energy.index.duplicated(keep='first')]

            # To'liq soatbay vaqt oralig'ini yaratish ('h' - kichik harf bilan FutureWarning oldini oladi)
            full_range = pd.date_range(start=energy.index.min(), end=energy.index.max(), freq='h')
            
            # Reindex va Interpolatsiya (Vaqt uzilishlarini to'ldirish)
            energy = energy.reindex(full_range)
            energy['PJME_MW'] = energy['PJME_MW'].interpolate(method='linear')
            energy = energy.reset_index().rename(columns={'index': 'Datetime'})
            logging_instance.info("Energy ma'lumotidagi vaqt uzilishlari va dublikatlar to'g'rilandi.")

            # 4. Temperature Data: Tozalash
            # Faqat Philadelphia ustunini olamiz
            philly_temp = temp[['datetime', 'Philadelphia']].copy()
            philly_temp = philly_temp.rename(columns={'datetime': 'Datetime', 'Philadelphia': 'Temp_K'})
            
            # Haroratdagi bo'shliqlarni 'Forward Fill' orqali to'ldirish (Leakage prevention)
            philly_temp = philly_temp.sort_values('Datetime')
            philly_temp['Temp_K'] = philly_temp['Temp_K'].ffill()
            logging_instance.info("Harorat ma'lumotidagi bo'shliqlar 'ffill' orqali to'ldirildi.")

            # 5. Birlashtirish (Merge) - Faqat ikkala ma'lumot kashishgan vaqtlar uchun
            combined_df = pd.merge(energy, philly_temp, on='Datetime', how='inner')
            logging_instance.info(f"Ma'lumotlar 'inner join' qilindi. Jami qatorlar: {len(combined_df)}")

            # 6. Saqlash
            output_path = os.path.join(self.output_dir, "combined_data.csv")
            combined_df.to_csv(output_path, index=False)
            logging_instance.info(f"Birlashtirilgan ma'lumot saqlandi: {output_path}")

            return combined_df

        except Exception as e:
            logging_instance.error(f"Ingestion bosqichida jiddiy xato: {str(e)}")
            raise e