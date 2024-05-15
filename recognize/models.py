from django.db import models

class Cart(models.Model):
    item_id = models.IntegerField()
    item_name = models.CharField(max_length=50)
    item_price = models.IntegerField()
    item_qty = models.IntegerField()
    



    @property
    def get_total(self):
        price = self.item_price * self.item_qty
        return price

    def __str__(self):
        return self.item_name

