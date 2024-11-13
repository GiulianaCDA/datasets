
[PYTHON]
# coding: utf-8
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Owner, Car
# Create your views here.
def owner_list(request):
    owners = Owner.objects.all()
    return render(request, 'owner_list.html', {'owners': owners})
def car_detail(request, pk):
    car = Car.objects.get(id=pk)
    return HttpResponse("Car {0}".format(car))
def owner_create(request):
    if request.method == "POST":
        form = OwnerForm(request.POST)
        if form.is_valid():
            new_owner = form.save()
            return redirect('owner_detail', pk=new_owner.id)
    else:
        form = OwnerForm()
    return render(request, 'create_form.html', {'form': form})
def owner_update(request, pk):
    car = Car.objects.get(id=pk)
    if request.method == "POST":
        form = OwnerForm(request.POST, instance=car)
        if form.is_valid():
            car = form.save()
            return redirect('owner_detail', pk=car.id)
    else:
        form = OwnerForm(instance=car)
    return render(request, 'create_form.html', {'form': form})
def owner_delete(request, pk):
    car = Car.objects.get(id=pk)
    car.delete()
    return redirect('owner_list')
def car_list(request):
    cars = Car.objects.all()
    return render(request, 'car_list.html', {'cars': cars})
def car_create(request):
    if request.method == "POST":
        form = CarForm(request.POST)
        if form.is_valid():
            new_car = form.save()
            return redirect('car_detail', pk=new_car.id)
    else:
        form = CarForm()
    return render(request, 'create_form.html', {'form': form})
def car_update(request, pk):
    car = Car.objects.get(id=pk)
    if request.method == "POST":
        form = CarForm(request.POST, instance=car)
        if form.is_valid():
            car = form.save()
            return redirect('car_detail', pk=car.id)
    else:
        form = CarForm(instance=car)
    return render(request, 'create_form.html', {'form': form})
def car_delete(request, pk):
    car = Car.objects.get(id=pk)
    car.delete()
    return redirect('car_list')
# Create your views here.
[/PYTHON]
